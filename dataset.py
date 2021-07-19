# PyTorch custom dataset for synthesizer midi parameters and sound samples.
# Author: Noah Mushkin, 07-18-2021

from os import listdir, path
from json import load
from math import sqrt
import PIL

import numpy as np
import torch
from torch.utils.data import Dataset

import minilogue


IMAGE_SIZE = 210

MEAN, STANDARD_DEV = -2.256989529720561e-05, 0.042319321922902144


class SynthSoundsDataset(Dataset):
    """Synth sound + parameter dataset."""

    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the json data files (with param + sample arrays).
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_list = list([f for f in listdir(root_dir) if '.json' in f])
        self.root_dir = root_dir
        self.mean = MEAN
        self.standard_dev = STANDARD_DEV

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        json_fname = path.join(self.root_dir, f'{idx + 1}.json')
        json_fp = open(json_fname, 'r')
        json_data = load(json_fp)
        json_fp.close()
        control_json = json_data['controls']
        controls = np.array([normalized_control(int(c), control_json[c]) for c in control_json])
        control_tensor = torch.from_numpy(controls).float()
        spectro_array = np.array(json_data['sample'][2000:]).reshape(1, 22000)
        sample_tensor = torch.from_numpy(spectro_array).float()

        return control_tensor, sample_tensor


def normalize_tensor(tensor, mean, std_dev):
    return (tensor - mean) / std_dev


def normalized_control(control_num, control_val):
    control_type = minilogue.CONTROL_TYPES.get(control_num)
    control_choices = minilogue.control_choices(control_type)
    return scale(control_val, [control_choices[0], control_choices[-1]], [0, 1])


# https://stackoverflow.com/questions/4154969/how-to-map-numbers-in-range-099-to-range-1-01-0/33127793
def scale(val, src, dst):
    """
    Scale the given value from the scale of src to the scale of dst.
    """
    return ((val - src[0]) / (src[1]-src[0])) * (dst[1]-dst[0]) + dst[0]


def calculate_data_stats(data_directory):
    data_files = listdir(data_directory)
    num_files = len(data_files)
    pixel_sum = 0
    squared_pixel_sum = 0
    pixel_count = IMAGE_SIZE * IMAGE_SIZE * num_files
    for fcount, fname in enumerate(data_files):
        with open(path.join(data_directory, fname)) as fp:
            json_data = load(fp)
            sample_array = np.array(json_data['sample'])
            pixel_sum += np.sum(sample_array)
            squared_pixel_sum += np.sum(sample_array ** 2)
        
        if (fcount % 100 == 0):
            print(round(fcount / num_files * 100)) 
    
    total_mean = pixel_sum / pixel_count
    total_variance = (squared_pixel_sum / pixel_count) - (total_mean ** 2)
    total_std = sqrt(total_variance)

    return total_mean, total_std
