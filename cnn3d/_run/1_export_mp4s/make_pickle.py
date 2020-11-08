#%%
import os
import cv2  # OpenCV
from glob import glob
from tqdm import tqdm
import pickle
import time
# 功能：对视频文件进行剪切。
# 剪切指定长度的视频，选择要裁剪的视频，选择开始时间点和停止时间点即可。
# 将处理后的视频保存为short+源文件名mp4文件

"""This file should be self-explanatory. It copies MP4s of faces from folders where they are stored according to the
original DFDC part they came in"""

i=1

SOURCE_DIR = "/data1/pbw/DFDC/cnn3d/data/whole_picture_by_real_fake"  # Used to locate metadata.json
MP4_DIR = "/data1/pbw/dfdc_dataset/video/train/"
OUT_DIR = "/data1/pbw/DFDC/cnn3d/data/whole_picture_by_real_fake"
#mp4paths = sorted(glob(os.path.join(SOURCE_DIR, "**/*.mp4")))
with open("/data1/pbw/DFDC/cnn3d/_run/1_export_mp4s/path.pk",'rb') as f:
    mp4paths=pickle.load(f)

