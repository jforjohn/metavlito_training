from torch.utils.data.sampler import Sampler
from benchmarks.utils.consts import CsvFieldLiteral
import numpy as np
import random


class BucketRandomSampler(Sampler):
  """
  Takes a dataset with 'buckets' property, cuts it into batch-sized chunks
  Drops the extra items, not fitting into exact batches
  Arguments:
  data_source (Dataset): a Dataset to sample from. Should have a cluster_indices property
  batch_size (int): a batch size that you would like to use later with Dataloader class
  shuffle (bool): whether to shuffle the data or not
  """

  def __init__(self, data, batch_size, shuffle=True, batch_type='variable'):
    self.data = data
    self.batch_type = batch_type
    self.batch_size = batch_size
    self.shuffle = shuffle
    # in order for the __len__ function to work we need
    # to initialize batch_lists
    self.batch_lists = self._createBatchBuckets()
    if self.data.buckets:
      print(self.data.subset)
      for ind in self.data.buckets:
        bucket = self.data.buckets[ind]
        print('Bucket index, size:', ind, len(bucket))
      
  def _batchFromBuckets(self, buckets):
    batch_lists = []
    data_remaining = True

    # find max of each bucket
    max_data = {}  
    for ind in buckets:
      bucket = buckets[ind]
      data_cols = CsvFieldLiteral.DATA.split(',')
      max_data[ind] = self.data.metadata[data_cols].loc[bucket].max(axis=0).values

      # randomize bucket members
      if self.shuffle:
        np.random.shuffle(buckets[ind])

    while data_remaining:
      ind = np.random.choice(list(buckets.keys()))
      bucket = buckets[ind]

      if len(bucket) <= self.batch_size:
        #batch_lists.append([ind, len(bucket), bucket])
        batch_lists.append(bucket.tolist())
        del buckets[ind]

      else:
        batch_bucket, buckets[ind] = bucket[:self.batch_size], bucket[self.batch_size:]
        #batch_lists.append([ind, batch_size, batch_cluster])
        batch_lists.append(batch_bucket.tolist())

      if not buckets:
        data_remaining = False
    
    return batch_lists
  
  def _batchFromIndex(self):
    # if buckets don't exist use the entire index
    # of the DataFrame
    index_list = self.data.metadata.index.tolist()
    if self.shuffle:
      np.random.shuffle(index_list)
    batch_lists = [index_list[i:i+self.batch_size] 
          for i in range(0, len(index_list), self.batch_size)]
    return batch_lists

  def _createBatchBuckets(self):
    buckets = self.data.buckets
    if not buckets:
      batch_lists = self._batchFromIndex()
    else:
      batch_lists = self._batchFromBuckets(buckets.copy())
    return batch_lists

  def unzip_list(self, lst):
      bucket_batch = lst
      if isinstance(lst[0], tuple) or isinstance(lst[0], list):
          if len(lst[0]) == 3:
              #print(42,lst[0])
              _, _, bucket_batch = zip(*lst)
      return bucket_batch
    
  def __iter__(self):
    self.batch_lists = self._createBatchBuckets()
    #lst = self.unzip_list(self.batch_lists)
    lst = self.batch_lists
    self.data.ds = self.data.metadata.loc[np.hstack(lst),:]
    return iter(lst)

  def __len__(self):
    return len(self.batch_lists)