#%%
import numpy as np 
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
import os

DIR="/data1/pbw/DFDC/cnn3d/useful_models/origin/"
paths=os.listdir(DIR)
#paths.sort(pat)
target_dict={}
for path in paths:
    # print(os.stat(DIR+path).st_mtime)
    #
    # print(path.split(".")[0])
    num_epoch=path.split("_")[1]

    target_dict[num_epoch[1:]]=path.split(".")[1]

#target_dict.sort()
print(target_dict)

length=len(target_dict)
x =np.arange(length)
real_y=[]
for i in x:
    fake_y =  float(target_dict[str(i)])/10000
    real_y.append(fake_y)

plt.title(DIR.split('/')[-2]) 
plt.xlabel("epoch_num") 
plt.ylabel("loss") 
plt.plot(x,real_y)
z=0.69314*np.ones(length)
plt.plot(x,z,linewidth=1,linestyle="--")
x_major_locator=MultipleLocator(1)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(0.1)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为10的倍数
plt.xlim(-0.5,20)
#把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
plt.ylim(-0.1,1)
#ax.patch.set_facecolor("gray")            #设置ax1区域背景颜色               

#ax.patch.set_alpha(0.5)                      #设置ax1区域背景颜色透明度       
#把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
#plt.figure(facecolor='blue',edgecolor='black')
filename=DIR.split('/')[-2]+'.png'
#print(filename)
plt.savefig(filename)
plt.show()

# %%
