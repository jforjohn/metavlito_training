from PIL import Image
import numpy as np

class FFTtransform(object):
  def __call__(self, img):
    img = np.array(img)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    
    return Image.fromarray(magnitude_spectrum.astype(np.uint8))
  
  def __repr__(self):
    return self.__class__.__name__ + '()'