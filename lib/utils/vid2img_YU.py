import cv2
import numpy as np
import glob
import os

img_array = []
rootPath = '/home/yu/Documents/data/Sanken/TestSample/Sep_19_Cam34/2'

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
for filename in os.listdir(rootPath):
    img = cv2.imread(os.path.join(rootPath, filename))
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('project.mp4', fourcc, 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

