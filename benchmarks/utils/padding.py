from torch import nn
import torch
from benchmarks.utils.tensorboard_writer import SummaryWriter
from benchmarks.utils.consts import Split


no_batches = 0

def padding_fn(batch, writer=None, subset='TRAIN'):
  global no_batches
  images = [item[0] for item in batch]
  labels = [item[1] for item in batch]
  max_height = max(img.shape[1] for img in images)
  max_width = max(img.shape[2] for img in images)
  max_area = max_height * max_width

  total_padding_area = 0
  padded_images = []
  for img in images:
    height, width = img.shape[1:]
    total_padding_area += max_area - height*width
    h_pad = max_height - height
    w_pad = max_width - width
    top_pad, bottom_pad = h_pad // 2, h_pad // 2
    left_pad, right_pad = w_pad // 2, w_pad // 2
    if h_pad % 2 != 0:
        bottom_pad = h_pad // 2 + 1
    if w_pad % 2 != 0:
        right_pad = w_pad // 2 + 1
    padding_tuple = (left_pad, right_pad, top_pad, bottom_pad)
    padded_img = nn.ZeroPad2d(padding_tuple)(img)
    #print('Padded Img', padded_img.size())
    padded_images.append(padded_img.expand(1, *padded_img.shape))
  
  #print('Max width, height + len_batch:', max_width, max_height, len(batch), flush=True)
  if writer:
    no_batches += 1
    print(subset, no_batches, total_padding_area, flush=True)
    writer[getattr(Split, subset)].add_scalar(writer.TPA, total_padding_area, global_step=no_batches)

  images_tensor = torch.cat(padded_images)
  labels_tensor = torch.LongTensor(labels)
  return images_tensor, labels_tensor