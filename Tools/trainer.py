import os 
import torch

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, \
                        convert_splitbn_model, convert_sync_batchnorm, model_parameters, set_fast_norm
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler

class TimmTrainer():
    device = 'cpu'
    optimizer = None
    model = None
    scheduler = None
    def __init__(self, config:object) -> None:
        self.config = config
        if config.use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
        
        self.model = self.get_model(config)
        self.optimizer = self.get_optimizer(config)
        self.lr_scheduler, self.num_epochs = self.get_scheduler(config)
        
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
        return model.to(self.device)

    def get_optimizer(self, args:object):
        optimizer = create_optimizer_v2(self.model **optimizer_kwargs(cfg=args))
        return optimizer

    def get_scheduler(self, args:object):
        lr_scheduler, num_epochs = create_scheduler(args, self.optimizer)
        return lr_scheduler, num_epochs

    def get_loader(self):
        pass

    def get_metrics(self):
        pass