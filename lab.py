import csv
from Datasets.satefarm import StateFarmDataset
from Configs.timm import get_config_debug

cfg, unparsed = get_config_debug('./Configs/debug_config.txt')

dataset = StateFarmDataset(cfg)
a, b = dataset[0]
print('hi')