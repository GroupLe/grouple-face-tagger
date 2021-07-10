from typing import NewType
import numpy as np

# 3 channel BGR cv2 image
Image = NewType('Image', np.ndarray)
