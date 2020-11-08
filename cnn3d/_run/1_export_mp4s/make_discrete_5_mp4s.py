#%%
import os
import cv2
from glob import glob
from tqdm import tqdm
import pickle
import sys
# 功能：对视频文件进行抽帧。
# 抽取指定间隔的帧做成视频。
# 将处理后的视频保存为源文件名mp4文件


i=int(sys.argv[1])

SOURCE_DIR = "/data1/pbw/dfdc_dataset/video/train/"
OUT_DIR = "/data1/pbw/DFDC/cnn3d/data/discrete_5/"



#%%
#mp4paths = sorted(glob(os.path.join(SOURCE_DIR, "**/*.mp4")))
with open('discrete_5.pk','rb') as f:
    mp4paths=pickle.load(f)
mp4paths=tuple(mp4paths)

#%%
#for filename in tqdm(mp4paths[0:1]):
for filename in tqdm(mp4paths[10000*(i-1):10000*i]):
    cap = cv2.VideoCapture(filename)  # 打开视频文件
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 获得视频文件的帧数
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获得视频文件的帧率
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 获得视频文件的帧宽
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 获得视频文件的帧高

    # 创建保存视频文件类对象
    #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUT_DIR+filename.split('/')[-2]+'/'+filename.split('/')[-1], fourcc, fps, (int(width), int(height)))

    #start = float(1)
    #stop = float(70)
    # 设置帧读取的开始位置
    #cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)  # 获得帧位置
    # print(pos)
    # print(frames)
#%%
    while (pos < frames):
        ret, frame = cap.read()  # 捕获一帧图像
        if pos%5==0:
            out.write(frame)  # 保存帧
        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        #print("after:",pos)
    #print("finished")
    cap.release()
    out.release()

