from Configs.timm import get_config, get_config_debug
from Tools.utils import fix_seed
from Tools.trainer import TimmTrainer
from termcolor import cprint
import timm
import argparse


def main(config:object):
    fix_seed(config.seed)
    trainer = TimmTrainer(config)
    trainer.train()


if __name__ == '__main__':
    cprint("Using TIMM as backend", 'blue', attrs=['bold'])
    parser = argparse.ArgumentParser('config parser')
    parser.add_argument('--config', type=str, default='./Configs/debug_config.txt')
    arg = parser.parse_args()
    config, unparsed = get_config_debug(arg.config)
    main(config)
