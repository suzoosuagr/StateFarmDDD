import os
from Tools.utils import get_files
from torch.utils.data import Dataset
from PIL import Image
import csv
from termcolor import cprint
import glob

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

str2label = lambda x: int(x[-1])

class StateFarmDataset(Dataset):
    
    def __init__(self, config, split:str='train', transform=None, target_transform=None) -> None:
        super(StateFarmDataset, self).__init__()
        self.config = config
        self.split = split
        self.data_root = os.path.join(config.data_root, 'train') # using data from official train folder
        self.val_subjects = config.val_subjects                  #      the split was done by weiheng.
        self.transform = transform
        self.target_transform = target_transform
        self.files = self._get_filepath(self.data_root)
        self.filepaths = self.parser_csv(config.driver_image_csv, self.data_root, self.val_subjects, split)

    def __getitem__(self, index):
        path, target = self.filepaths[index]
        sample = pil_loader(path)
        if self.transform is not None:
            if isinstance(self.transform, tuple):
                for trans in self.transform:
                    sample = trans(sample)
            else:
                sample = self.transform(sample)
        if self.target_transform is not None:
            if isinstance(self.target_transform, tuple):
                for trans in self.target_transform:
                    target = trans(target)
            else:
                sample = self.target_transform(sample)

        return sample, target

    def _get_filepath(self, data_root):
        assert os.path.isdir(data_root)
        files = glob.glob(data_root + '/*/*.jpg')
        return files
        
    def parser_csv(self, csvfile, data_root, val_subjects, split):
        file_list = []
        invalid_list = []
        with open(csvfile) as f:
            lines = csv.reader(f)
            for idx, line in enumerate(lines):
                if idx == 0:
                    continue
                else:
                    subject, classname, filename = line
                    label = str2label(classname)
                    if (split == 'train' and subject not in val_subjects) or \
                       (split == 'test' and subject in val_subjects):
                        filepath = os.path.join(data_root, classname, filename)
                        if filepath in self.files:
                            file_list.append((filepath, label))
                        else:
                            invalid_list.append(filename)
        cprint(f" -> Parse {split}: {len(file_list)} images", 'cyan')
        cprint(f"    -> found {len(invalid_list)} invalid images", 'yellow')
        return file_list

    def __len__(self):
        # return len(self.filepaths)
        return 128
