import numpy as np
from PIL import Image

import os

folder_images = ('./GTSRB/images/34')
target = ('./GTSRB/images/34_new_mirror')

images = os.listdir(folder_images)


for x in images:
        
    original_img = Image.open(folder_images+"/" + x)
    
    # Flip the original image vertically
    vertical_img = original_img.transpose(method=Image.FLIP_LEFT_RIGHT)
    vertical_img.save(target+"/m"+x)
    
    # close all our files object
    original_img.close()
    vertical_img.close()