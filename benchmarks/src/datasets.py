import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from benchmarks.utils.consts import Split, CsvFieldLiteral
from benchmarks.utils import DynamicResize, GammaCorrection, FFTtransform, DynamicCrop
try:
    import cupy as cp
except ImportError:
    print('There is no CuPy library')
    cp = np

class ClassificationDataset(Dataset):
    """Image classification dataset."""

    def __init__(self, subset, csv_path,
                data_folder, buckets_files,
                config_training):
        """
        Args:
            subset (string): Subset of the dataset to load (either "train", "validation" or "test").
                on a sample.
        """
        #assert size[0] < 256 and size[1] < 256
        assert subset not in [attr for attr in dir(Split) if not attr.startswith('__')]
        self.csv_path = csv_path
        self.data_path = data_folder
        self.subset = subset

        self.buckets_files = buckets_files
        self.buckets = self._get_buckets()
        
        self.config_training = config_training
        
        self.transform = transforms.Compose(self._get_transforms_list())
        self.labels2idx, self.idx2labels, self.metadata = self._get_metadata()

    def _get_transforms_list(self):
        padding = self.config_training.getboolean('padding')
        dynamic_crop_ratio = self.config_training.getfloat('dynamic_crop_ratio')

        try:
            downratio = self.config_training.getfloat('down_ratio', 0.5)
        except ValueError:
            downratio = 1.0

        resize_h = self.config_training.getint('resize_h', 256)
        resize_w = self.config_training.getint('resize_w', 256)

        dr_max_size = self.config_training.getint('down_ratio_max', 800)
        dr_min_size = self.config_training.getint('down_ratio_min', 400)

        return_transform = [DynamicResize(
                                resize_h,
                                resize_w,
                                downratio,
                                dr_min_size,
                                dr_max_size
                            )]

        gamma = self.config_training.getfloat('g_correction', 1.0)
        if gamma > 0.0:    
            return_transform.append(GammaCorrection(gamma))

        fft = self.config_training.getboolean('fft')
        if fft:
            return_transform.append(FFTtransform())

        if not padding or dynamic_crop_ratio:

            crop_h = self.config_training.getint('crop_h', 0)
            crop_w = self.config_training.getint('crop_w', 0)
            return_transform.append(
                DynamicCrop(dynamic_crop_ratio, self.subset,
                            crop_h, crop_w)
            )

        return_transform.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        print('%s transformations:' %(self.subset))
        print(return_transform)
        return return_transform

    def _get_buckets(self):
        filepath = self.buckets_files.get(self.subset, None)
        buckets = None
        self.bucketing = False
        if filepath:
            try:
                buckets_file = np.load(filepath)
                buckets = buckets_file.item()
                self.bucketing = True
            except FileExistsError:
                print('%s path does not exist' %(filepath))
        return buckets

    def _get_metadata(self):
        df = pd.read_csv(self.csv_path)
        if CsvFieldLiteral.OBJECT_ID:
            df = df.set_index(CsvFieldLiteral.OBJECT_ID)
        labels2idx = {}
        idx2labels = {}
        for idx, label in enumerate(df["Label"].unique()):
            labels2idx[label] = idx
            idx2labels[idx] = label
        metadata = df[df['Split'] == self.subset]
        return labels2idx, idx2labels, metadata

    def get_idx2labels(self):
        return self.idx2labels

    def _getBucketItem(self, idx):
        images_lst = []
        targets_lst = []
        
        for i in idx:
            img_name = os.path.join(self.data_path,
                    self.metadata.at[i, CsvFieldLiteral.IMAGE_FILE])
            image = Image.open(img_name).convert('RGB')
            label = self.metadata.at[i, CsvFieldLiteral.TARGET]
            label_idx = self.labels2idx[label]
            image = self.transform(image)
            images_lst.append(image)
            targets_lst.append(label_idx)
        
        ret = (images_lst, targets_lst)
        return ret   

    '''def fft_transformation(self, img):
        img = cp.array(img.convert('LA'))

        f = np.fft.fft2(img[:,:,0])
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        
        #np_magnitude_spectrum = cp.asnumpy(magnitude_spectrum)
        #del img, f, fshift, magnitude_spectrum
        return Image.fromarray(
            np.transpose(
                np.array([magnitude_spectrum.astype(np.uint8), img[:,:,1]]),(1,2,0)))'''

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        '''if self.buckets:
            # if bucketing is True then we get 
            # a list with all the idx of a batch
            return self._getBucketItem(idx)
        else:'''
        img_name = os.path.join(self.data_path,
                            self.metadata.at[idx, CsvFieldLiteral.IMAGE_FILE])
        image = Image.open(img_name).convert('RGB')

        label = self.metadata.at[idx, CsvFieldLiteral.TARGET]
        label_idx = self.labels2idx[label]
        image = self.transform(image)
        #print('ds', image.size())

        return image, label_idx
