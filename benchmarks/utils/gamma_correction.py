class GammaCorrection(object):
  def __init__(self, gamma, gain=1):
    self.gamma = gamma
    self.gain = gain
  
  def __call__(self, img):
      input_mode = img.mode
      img = img.convert('RGB')
      gamma_map = [255 * self.gain * pow(ele / 255., self.gamma) for ele in range(256)] * 3
      img = img.point(gamma_map)  # use PIL's point-function to accelerate this part

      img = img.convert(input_mode)
      return img
  
  def __repr__(self):
      return self.__class__.__name__ + '(gamma={0}, gain={1})'.format(self.gamma, self.gain)