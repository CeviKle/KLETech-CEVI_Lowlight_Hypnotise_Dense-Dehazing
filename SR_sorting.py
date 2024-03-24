import os 
import cv2
import shutil


source_path = '/home/pegasus/DATA/Nikhil/LLIE/SR/Mix/LR_renamed/'
dest_path = '/home/pegasus/DATA/Nikhil/LLIE/SR/FLICKR/train/LR/'

for img in os.listdir(source_path):
    if 'FLICKR' in img:
        shutil.copy(os.path.join(source_path + img), os.path.join(dest_path + img))
        print(img)
