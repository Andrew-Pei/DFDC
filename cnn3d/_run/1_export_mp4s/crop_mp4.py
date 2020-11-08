#%%
import os
import cv2  # OpenCV
from glob import glob
from tqdm import tqdm
import pickle

# 本程序为实验1 whole picture的数据集加载提供支持，由于原本数据集太大，加载费时。
# 因此本程序将实际需要的视频片段进行裁剪。
# 功能：对视频文件进行剪切。
# 剪切指定长度的视频，选择要裁剪的视频，选择开始时间点和停止时间点即可。
# 将处理后的视频保存为short+源文件名mp4文件

"""This file should be self-explanatory. It copies MP4s of faces from folders where they are stored according to the
original DFDC part they came in"""

i=6

SOURCE_DIR = "/data1/pbw/DFDC/cnn3d/data/whole_picture_by_real_fake"  # Used to locate metadata.json
MP4_DIR = "/data1/pbw/dfdc_dataset/video/train/"
OUT_DIR = "/data1/pbw/DFDC/cnn3d/data/whole_picture_by_real_fake"
#mp4paths = sorted(glob(os.path.join(SOURCE_DIR, "**/*.mp4")))
with open('path.pk','rb') as f:
    mp4paths=pickle.load(f)


for filename in tqdm(mp4paths[10000*(i-1):10000*i]):
    cap = cv2.VideoCapture(filename)  # 打开视频文件
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 获得视频文件的帧数
    if frames==0.0 or frames==70.0:
        #print("yes")
        continue
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获得视频文件的帧率
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 获得视频文件的帧宽
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 获得视频文件的帧高

    # 创建保存视频文件类对象
    #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUT_DIR+'/'+filename.split('/')[-2]+'/short'+filename.split('/')[-1], fourcc, fps, (int(width), int(height)))

    start = float(1)
    stop = float(70)
    # 设置帧读取的开始位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)  # 获得帧位置
    while (pos <= stop):
        ret, frame = cap.read()  # 捕获一帧图像
        out.write(frame)  # 保存帧
        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

    cap.release()
    out.release()
    #删除原视频

    os.remove(filename)
    #更名
    os.rename(OUT_DIR+'/'+filename.split('/')[-2]+'/short'+filename.split('/')[-1],filename)


