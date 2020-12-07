from PIL import Image
from torchvision.transforms import Resize

class DynamicResize(object):
  def __init__(self, resize_h, resize_w,
               ratio, min_size, max_size):
    self.ratio = ratio
    self.resize_h = resize_h
    self.resize_w = resize_w
    self.min_size = min_size
    self.max_size = max_size
  
  def __call__(self, img):
    if self.resize_h == 0 or self.resize_w == 0:
      width, height = img.size
      ratio = self.ratio

      max_wh = max(width, height)
      min_wh = min(width, height)
      if min_wh*ratio < self.min_size and max_wh*ratio < self.max_size:
        ratio = self.min_size/min_wh
        
      if width*ratio > self.max_size or height*ratio > self.max_size:
        ratio = self.max_size/max_wh
      
      res_w, res_h = round(width*ratio), round(height*ratio)
      res_img = img.resize((res_w, res_h), Image.ANTIALIAS)
    
    else:
      res_img = Resize((self.resize_h, self.resize_w))(img)

    return res_img
  
  def __repr__(self):
    if self.resize_h == 0 or self.resize_w == 0:
      arguments = '(downratio={0}, MinMaxSize=[{1},{2}])'.format(self.ratio, self.min_size, self.max_size)

    else:
      arguments = '({0}, {1})'.format(self.resize_h, self.resize_w)
    return self.__class__.__name__ + arguments