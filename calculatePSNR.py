import os 
import numpy 
import cv2 
import pandas as pd 
from tqdm import tqdm 

from skimage import metrics
import natsort



restoredDir = '/home/cvg-ws05/msi_up/LLIE/archives/test_report1/generated/'

groundtruthDir = '/home/cvg-ws05/msi_up/LLIE/archives/test_report1/gt/'
restoredList = natsort.natsorted(os.listdir(restoredDir))
groundtruthList = natsort.natsorted(os.listdir(groundtruthDir))

print(len(restoredList),len(groundtruthList))

for i in range(len(restoredList)):
	try:
		restoredImage = cv2.imread(restoredDir + restoredList[i])
		grountruthImage = cv2.imread(groundtruthDir + groundtruthList[i])
		psnr = metrics.peak_signal_noise_ratio(restoredImage,grountruthImage)
		SSIM = metrics.structural_similarity(grountruthImage,restoredImage,multichannel = True)
		print(restoredList[i] + ' ' + groundtruthList[i] + ' PSNR = ' + str(psnr) + ' ,SSIM = ' + str(SSIM))
	except:
		print(groundtruthList[i],restoredList[i])

