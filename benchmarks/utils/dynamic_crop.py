from benchmarks.utils.consts import Split
from torchvision.transforms import RandomCrop, CenterCrop

class DynamicCrop(object):
  # gives priority if the exact size of the
  # cropped images is given
  def __init__(self, ratio, subset, crop_h, crop_w):
    self.ratio = ratio
    self.subset = subset
    self.crop_h = crop_h
    self.crop_w = crop_w
    
  def __call__(self, img):
    width, height = img.size
    
    # TODO: add checks eg ratio to be (0,1)
    if self.crop_h == 0 or self.crop_w == 0:
      crop_h, crop_w = round(height*self.ratio), round(width*self.ratio)
    else:
      crop_h, crop_w = self.crop_h, self.crop_w

    if self.subset == Split.TRAIN:
      crop = RandomCrop((crop_h, crop_w))
    else:
      crop = CenterCrop((crop_h, crop_w))
    
    return crop(img)
  
  def __repr__(self):
    if self.subset == Split.TRAIN:
      func = 'Random'
    else:
      func = 'Center'
    
    if self.crop_w == 0 or self.crop_h == 0:
      arguments = '(ratio={0})'.format(self.ratio)
    else:
      arguments = '({0}, {1})'.format(self.crop_h, self.crop_w)

    return func + self.__class__.__name__ + arguments