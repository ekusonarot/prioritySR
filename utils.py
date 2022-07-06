import numpy as np
import cv2

class ConvertYcbcr2Bgr:
  def __init__(self):
    pass

  def __call__(self, img):
    r = 298.082 * img[..., 0] / 256. + 408.583 * img[..., 2] / 256. - 222.921
    g = 298.082 * img[..., 0] / 256. - 100.291 * img[..., 1] / 256. - 208.120 * img[..., 2] / 256. + 135.576
    b = 298.082 * img[..., 0] / 256. + 516.412 * img[..., 1] / 256. - 276.836
    return np.array([b, g, r]).transpose([1, 2, 0])

class ConvertBgr2Ycbcr:
  def __init__(self):
    pass

  def __call__(self, img):
    y = 16. + (64.738 * img[..., 2] + 129.057 * img[..., 1] + 25.064 * img[..., 0]) / 256.
    cb = 128. + (-37.945 * img[..., 2] - 74.494 * img[..., 1] + 112.439 * img[..., 0]) / 256.
    cr = 128. + (112.439 * img[..., 2] - 94.154 * img[..., 1] - 18.285 * img[..., 0]) / 256.
    return np.array([y, cb, cr]).transpose([1, 2, 0])

class ConvertBgr2Y:
  def __init__(self):
    pass

  def __call__(self, img):
    img = np.array(img)
    return (16. + (64.738 * img[..., 2] + 129.057 * img[..., 1] + 25.064 * img[..., 0]) / 256.) / 255.

class Resize:
  def __init__(self, fx=1, fy=1):
    self.fx = fx
    self.fy = fy

  def __call__(self, img):
    img = np.array(img)
    return cv2.resize(img, None, fx=self.fx, fy=self.fy, interpolation=cv2.INTER_NEAREST)