import os
import numpy as np
from torch.utils.data import DataLoader
from benchmarks.utils.Sampler import BucketRandomSampler
from benchmarks.utils.padding import padding_fn

from benchmarks.utils.consts import Split

class InputPipeline(object):

    def __init__(self, datasets_list,
                 batch_size=1,
                 writer=None, num_workers=None,
                 seed=None):
        if num_workers is None:
            try:
                num_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
            except KeyError:
                num_workers = 8
        print('Number of workers:', num_workers)

        self.seed = seed
        self.dataloaders = {}

        for ds in datasets_list:
            shuffle = ds.subset == Split.TRAIN
                       
            batch_sampler = BucketRandomSampler(ds, batch_size, shuffle)
                        
            dl_kwargs = {
                'batch_sampler': batch_sampler,
                'num_workers': num_workers
            }
            if ds.config_training.getboolean('padding'):
                if ds.config_training.getboolean('tpa_track'):

                    writer_summary = writer
                else:
                    writer_summary = None
                
                if ds.subset == Split.TRAIN:
                    dl_kwargs.update({
                        'collate_fn': lambda x: padding_fn(x, writer=writer_summary, subset='TRAIN'),
                    })
                elif ds.subset == Split.VAL:
                    dl_kwargs.update({
                        'collate_fn': lambda x: padding_fn(x, writer=writer_summary, subset='VAL'),
                    })
                else:
                    dl_kwargs.update({
                        'collate_fn': lambda x: padding_fn(x, writer=writer_summary, subset='TEST'),
                    })
                
                
            dl = self.get_dataloader(
                ds,
                **dl_kwargs
            )
            self.dataloaders[ds.subset] = dl

    def get_dataloader(self, *args, **kwargs):
        if self.seed:
            kwargs['worker_init_fn'] = lambda: np.random.seed(self.seed)
        return DataLoader(*args, **kwargs)

    def __getitem__(self, subset):
        try:
            return self.dataloaders[subset]
        except KeyError:
            return None
