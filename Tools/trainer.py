import os
import torch
import torch.nn as nn
from termcolor import cprint
import time
import logging
import torch
import torchvision.utils
from collections import OrderedDict

from timm import utils
from timm.data import create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, \
                        convert_splitbn_model, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler

from Datasets.satefarm import StateFarmDataset
from Tools.io import ensure
from Tools.metrics import ConfMat
from contextlib import suppress

_logger = logging.getLogger('train')
class TimmTrainer():
    device = 'cpu'
    optimizer = None
    model = None
    scheduler = None
    num_aug_splits = 0
    saver = None
    def __init__(self, config:object) -> None:
        self.config = config
        if config.aug_splits > 0:
            assert config.aug_splits > 1
            self.num_aug_splits = config.aug_splits
        config.prefetcher = not config.no_prefetcher
        self.output_dir = os.path.join(config.output, config.experiment)
        if config.use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
            cprint("[*] Using CUDA", 'blue')
       
        self.model = self.get_model(config)
        self.optimizer = self.get_optimizer(config)
        self.lr_scheduler, self.num_epochs = self.get_scheduler(config)
        cprint(f"[*] load data from    {config.data_root}", 'blue')        
        self.loader_train, self.loader_test, self.mixup_fn = self.get_loader(config)
        self.train_loss_fn, self.validate_loss_fn = self.get_loss(config)
        self.saver = self.get_ckpt_saver(config)
        self.metric = self.get_metric(config)

        ensure(os.path.join(config.output, config.experiment))
        cprint(f"[*] ckpt will save to {os.path.join(config.output, config.experiment)}", "blue")
        self.config = config

    def get_model(self, args:object):
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            drop_rate=args.drop,
            global_pool=args.gp,
            bn_momentum=args.bn_momentum,
            bn_eps=args.bn_eps,
            scriptable=args.torchscript,
            checkpoint_path=args.initial_checkpoint)
        cprint(f"[*] Using {args.model} | Pretrained: {args.pretrained} | num_classes: {args.num_classes}")
        return model.to(self.device)

    def get_optimizer(self, args:object):
        optimizer = create_optimizer_v2(self.model, **optimizer_kwargs(cfg=args))
        return optimizer

    def get_scheduler(self, args:object):
        lr_scheduler, num_epochs = create_scheduler(args, self.optimizer)
        return lr_scheduler, num_epochs

    def get_loader(self, args:object):
        # set up mixup
        collate_fn = None
        mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            mixup_args = dict(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.num_classes)
            if args.prefetcher:
                assert not self.num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
                collate_fn = FastCollateMixup(**mixup_args)
            else:
                mixup_fn = Mixup(**mixup_args)

        data_config = resolve_data_config(vars(args), model=self.model, verbose=True)
        # -- dataset
        dataset_train = StateFarmDataset(args, split='train')
        dataset_test = StateFarmDataset(args, split='test')
        
        # create data loaders w/ augmentation pipeiine
        train_interpolation = args.train_interpolation
        if args.no_aug or not train_interpolation:
            train_interpolation = data_config['interpolation']
        loader_train = create_loader(
            dataset_train,
            input_size=data_config['input_size'],
            batch_size=args.batch_size,
            is_training=True,
            use_prefetcher=args.prefetcher,
            no_aug=args.no_aug,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            re_split=args.resplit,
            scale=args.scale,
            ratio=args.ratio,
            hflip=args.hflip,
            vflip=args.vflip,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            num_aug_repeats=args.aug_repeats,
            num_aug_splits=self.num_aug_splits,
            interpolation=train_interpolation,
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            # distributed=args.distributed,
            collate_fn=collate_fn,
            pin_memory=args.pin_mem,
            use_multi_epochs_loader=args.use_multi_epochs_loader,
            worker_seeding=args.worker_seeding,  
        )

        loader_eval = create_loader(
            dataset_test,
            input_size=data_config['input_size'],
            batch_size=args.validation_batch_size or args.batch_size,
            is_training=False,
            use_prefetcher=args.prefetcher,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            # distributed=args.distributed,
            crop_pct=data_config['crop_pct'],
            pin_memory=args.pin_mem,
        )

        return loader_train, loader_eval, mixup_fn

    def get_loss(self, args:object):
        # setup loss function
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if args.jsd_loss:
            assert self.num_aug_splits > 1  # JSD only valid with aug splits set
            train_loss_fn = JsdCrossEntropy(num_splits=self.num_aug_splits, smoothing=args.smoothing)
        elif mixup_active:
            # smoothing is handled with mixup target transform which outputs sparse, soft targets
            if args.bce_loss:
                train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
            else:
                train_loss_fn = SoftTargetCrossEntropy()
        elif args.smoothing:
            if args.bce_loss:
                train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
            else:
                train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            train_loss_fn = nn.CrossEntropyLoss()
        train_loss_fn = train_loss_fn.cuda()
        validate_loss_fn = nn.CrossEntropyLoss().cuda()
        return train_loss_fn, validate_loss_fn

    def get_ckpt_saver(self, args:object):
        decreassing = True if args.eval_metric == 'loss' else False
        saver = utils.CheckpointSaver(
            model=self.model, optimizer=self.optimizer, checkpoint_prefix='ckpt',
                checkpoint_dir=os.path.join(args.output, args.experiment),
                decreasing=decreassing, 
                max_history=3 
        )
        return saver

    def get_metric(self, args:object):
        metric = ConfMat(args.num_classes)
        return metric

    def train(self):
        patience = 0
        best_metric = None
        best_epoch = None
        min_loss = 1000

        for epoch in range(self.config.start_epoch, self.config.epochs):
            if patience > self.config.train_patience:
                cprint("OUT OF PATIENCE", color='red', on_color='on_white', attrs=['bold', 'blink']) 
                cprint(f"Epoch {epoch}/{self.config.epochs}", color='yellow')
            train_metrics = self.train_one_epoch(
                epoch, self.model, self.loader_train, self.optimizer, self.train_loss_fn, self.config,
                lr_scheduler=self.lr_scheduler, saver=self.saver, output_dir=self.output_dir,
                 mixup_fn=self.mixup_fn
            )
            eval_metrics = self.validate(self.model, self.loader_test, self.validate_loss_fn, self.config)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch+1, eval_metrics[self.config.eval_metric])

            if self.output_dir is not None:
                utils.update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(self.output_dir, 'summary.csv'),
                    write_header=best_metric is None)

            
            save_metric = eval_metrics[self.config.eval_metric]
            best_metric, best_epoch = self.saver.save_checkpoint(epoch, metric=save_metric)
            if best_epoch is not None:
                if eval_metrics['loss'] < min_loss:
                    patience = 0
                    min_loss = eval_metrics['loss']
                else:
                    patience += 1


    def train_one_epoch(self,
            epoch, model, loader, optimizer, loss_fn, args,
            lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
            loss_scaler=None, model_ema=None, mixup_fn=None):

        if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
            if args.prefetcher and loader.mixup_enabled:
                loader.mixup_enabled = False
            elif mixup_fn is not None:
                mixup_fn.mixup_enabled = False

        second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        batch_time_m = utils.AverageMeter()
        data_time_m = utils.AverageMeter()
        losses_m = utils.AverageMeter()

        model.train()

        end = time.time()
        last_idx = len(loader) - 1
        num_updates = epoch * len(loader)
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            data_time_m.update(time.time() - end)
            if not args.prefetcher:
                input, target = input.cuda(), target.cuda()
                if mixup_fn is not None:
                    input, target = mixup_fn(input, target)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
                loss = loss_fn(output, target)

           
            losses_m.update(loss.item(), input.size(0))

            optimizer.zero_grad()
            if loss_scaler is not None:
                loss_scaler(
                    loss, optimizer,
                    clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                    parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                    create_graph=second_order)
            else:
                loss.backward(create_graph=second_order)
                if args.clip_grad is not None:
                    utils.dispatch_clip_grad(
                        model_parameters(model, exclude_head='agc' in args.clip_mode),
                        value=args.clip_grad, mode=args.clip_mode)
                optimizer.step()

            if model_ema is not None:
                model_ema.update(model)

            torch.cuda.synchronize()
            num_updates += 1
            batch_time_m.update(time.time() - end)
            if last_batch or batch_idx % args.log_interval == 0:
                lrl = [param_group['lr'] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                # if args.distributed:
                #     reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                #     losses_m.update(reduced_loss.item(), input.size(0))

                if args.local_rank == 0:
                    cprint(
                        'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                        'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                        '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                        'LR: {lr:.3e}  '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            epoch,
                            batch_idx, len(loader),
                            100. * batch_idx / last_idx,
                            loss=losses_m,
                            batch_time=batch_time_m,
                            rate=input.size(0) / batch_time_m.val,
                            rate_avg=input.size(0) / batch_time_m.avg,
                            lr=lr,
                            data_time=data_time_m))

                    if args.save_images and output_dir:
                        torchvision.utils.save_image(
                            input,
                            os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                            padding=0,
                            normalize=True)

            if saver is not None and args.recovery_interval and (
                    last_batch or (batch_idx + 1) % args.recovery_interval == 0):
                saver.save_recovery(epoch, batch_idx=batch_idx)

            if lr_scheduler is not None:
                lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

            end = time.time()
            # end for

        if hasattr(optimizer, 'sync_lookahead'):
            optimizer.sync_lookahead()

        return OrderedDict([('loss', losses_m.avg)])


    def validate(self, model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
        batch_time_m = utils.AverageMeter()
        losses_m = utils.AverageMeter()
        top1_m = utils.AverageMeter()
        top3_m = utils.AverageMeter()
        top5_m = utils.AverageMeter()

        model.eval()

        end = time.time()
        last_idx = len(loader) - 1
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(loader):
                last_batch = batch_idx == last_idx
                if not args.prefetcher:
                    input = input.cuda()
                    target = target.cuda()
                if args.channels_last:
                    input = input.contiguous(memory_format=torch.channels_last)

                with amp_autocast():
                    output = model(input)
                if isinstance(output, (tuple, list)):
                    output = output[0]

                # augmentation reduction
                reduce_factor = args.tta
                if reduce_factor > 1:
                    output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                    target = target[0:target.size(0):reduce_factor]

                loss = loss_fn(output, target)
                acc1, acc3, acc5 = utils.accuracy(output, target, topk=(1, 3, 5))

                
                reduced_loss = loss.data

                torch.cuda.synchronize()

                losses_m.update(reduced_loss.item(), input.size(0))
                top1_m.update(acc1.item(), output.size(0))
                top3_m.update(acc3.item(), output.size(0))
                top5_m.update(acc5.item(), output.size(0))

                batch_time_m.update(time.time() - end)
                end = time.time()
                if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                    log_name = 'Test' + log_suffix
                    cprint(
                        '{0}: [{1:>4d}/{2}]  '
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                        'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                        'Acc@3: {top3.val:>7.4f} ({top3.avg:>7.4f})  '
                        'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                            log_name, batch_idx, last_idx, batch_time=batch_time_m,
                            loss=losses_m, top1=top1_m, top3=top3_m, top5=top5_m), color='green')

        metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top3', top3_m.avg), ('top5', top5_m.avg)])

        return metrics