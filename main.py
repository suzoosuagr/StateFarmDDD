from Configs.timm import get_config
from Tools.utils import fix_seed
from Tools.trainer import TimmTrainer
from termcolor import cprint


def main(config:object):
    fix_seed(config.seed)

    trainer = TimmTrainer(config)
    trainer.get_loader()


if __name__ == '__main__':
    config, unparsed = get_config()
    cprint(f"{len(unparsed)} arguments used default values")
    main(config)
