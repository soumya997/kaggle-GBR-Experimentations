#!/usr/bin/env python
# coding: utf-8

# ## Yolov5 high resolution training
# 
# ### Major modification
# * img=3600
# * mixup=0.5
# * fliplr: 0.5
# 
# ### Hardware to reproduce
# * RTX3090

# ### Training Log:
# > ```
# version=1
# img_size:3584,bs2,e11,[yolov5s6] 
# Fold: video2[validation]
# Labels: only GT
# ```

# In[1]:


get_ipython().system('nvidia-smi')


# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from shutil import copyfile

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# In[3]:


# train = pd.read_csv('../input/tensorflow-great-barrier-reef/train.csv')
# train['pos'] = train.annotations != '[]'


# In[4]:


df = pd.read_csv("../input/tensorflow-great-barrier-reef/train.csv")

# Turn annotations from strings into lists of dictionaries
df['annotations1'] = df['annotations'].apply(eval)

# Create the image path for the row
df['image_path'] = "video_" + df['video_id'].astype(str) + "/" + df['video_frame'].astype(str) + ".jpg"

length = lambda x: len(x) 

df["no_of_bbox"] = df["annotations1"].apply(length)

df.head(5)


# In[5]:


val_df = df[df["video_id"]==0][df["no_of_bbox"]>0]
val_df.shape


# In[6]:


train_df = df[df["video_id"]!=0][df["no_of_bbox"]>0]
train_df.shape


# In[7]:


get_ipython().system('mkdir -p ./yolo_data/fold0/images/val')
get_ipython().system('mkdir -p ./yolo_data/fold0/images/train')

get_ipython().system('mkdir -p ./yolo_data/fold0/labels/val')
get_ipython().system('mkdir -p ./yolo_data/fold0/labels/train')


# In[8]:


fold = 0

annos = []
for i, x in val_df.iterrows():
#     if x.video_id == fold:
#         if x.pos:
    mode = 'val'
#     else:
#         # train
#         mode = 'train'
#         if not x.pos: continue
        # val
    copyfile(f'../input/tensorflow-great-barrier-reef/train_images/video_{x.video_id}/{x.video_frame}.jpg',
                f'./yolo_data/fold{fold}/images/{mode}/{x.image_id}.jpg')
#     if not x.pos:
#         continue
    r = ''
    anno = eval(x.annotations)
    for an in anno:
#            annos.append(an)
        r += '0 {} {} {} {}\n'.format((an['x'] + an['width'] / 2) / 1280,
                                        (an['y'] + an['height'] / 2) / 720,
                                        an['width'] / 1280, an['height'] / 720)
    with open(f'./yolo_data/fold{fold}/labels/{mode}/{x.image_id}.txt', 'w') as fp:
        fp.write(r)


# In[9]:


fold = 0

annos = []
for i, x in train_df.iterrows():
#     if x.video_id == fold:
#         if x.pos:
    mode = 'train'
#     else:
#         # train
#         mode = 'train'
#         if not x.pos: continue
        # val
    copyfile(f'../input/tensorflow-great-barrier-reef/train_images/video_{x.video_id}/{x.video_frame}.jpg',
                f'./yolo_data/fold{fold}/images/{mode}/{x.image_id}.jpg')
#     if not x.pos:
#         continue
    r = ''
    anno = eval(x.annotations)
    for an in anno:
#            annos.append(an)
        r += '0 {} {} {} {}\n'.format((an['x'] + an['width'] / 2) / 1280,
                                        (an['y'] + an['height'] / 2) / 720,
                                        an['width'] / 1280, an['height'] / 720)
    with open(f'./yolo_data/fold{fold}/labels/{mode}/{x.image_id}.txt', 'w') as fp:
        fp.write(r)


# In[ ]:





# In[10]:


# fold = 1

# annos = []
# for i, x in train.iterrows():
#     if x.video_id == fold:
#         if x.pos:
#             mode = 'val'
#     else:
#         # train
#         mode = 'train'
#         if not x.pos: continue
#         # val
#     copyfile(f'../input/tensorflow-great-barrier-reef/train_images/video_{x.video_id}/{x.video_frame}.jpg',
#                 f'./yolo_data/fold{fold}/images/{mode}/{x.image_id}.jpg')
#     if not x.pos:
#         continue
#     r = ''
#     anno = eval(x.annotations)
#     for an in anno:
# #            annos.append(an)
#         r += '0 {} {} {} {}\n'.format((an['x'] + an['width'] / 2) / 1280,
#                                         (an['y'] + an['height'] / 2) / 720,
#                                         an['width'] / 1280, an['height'] / 720)
#     with open(f'./yolo_data/fold{fold}/labels/{mode}/{x.image_id}.txt', 'w') as fp:
#         fp.write(r)


# In[11]:


import os
len(os.listdir("./yolo_data/fold0/labels/train"))


# In[12]:


hyps = '''
# YOLOv5 by Ultralytics, GPL-3.0 license
# Hyperparameters for COCO training from scratch
# python train.py --batch 40 --cfg yolov5m.yaml --weights '' --data coco.yaml --img 640 --epochs 300
# See tutorials for hyperparameter evolution https://github.com/ultralytics/yolov5#tutorials

lr0: 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
lrf: 0.1  # final OneCycleLR learning rate (lr0 * lrf)
momentum: 0.937  # SGD momentum/Adam beta1
weight_decay: 0.0005  # optimizer weight decay 5e-4
warmup_epochs: 3.0  # warmup epochs (fractions ok)
warmup_momentum: 0.8  # warmup initial momentum
warmup_bias_lr: 0.1  # warmup initial bias lr
box: 0.05  # box loss gain
cls: 0.5  # cls loss gain
cls_pw: 1.0  # cls BCELoss positive_weight
obj: 1.0  # obj loss gain (scale with pixels)
obj_pw: 1.0  # obj BCELoss positive_weight
iou_t: 0.20  # IoU training threshold
anchor_t: 4.0  # anchor-multiple threshold
# anchors: 3  # anchors per output layer (0 to ignore)
fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5)
hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
hsv_v: 0.4  # image HSV-Value augmentation (fraction)
degrees: 0.0  # image rotation (+/- deg)
translate: 0.1  # image translation (+/- fraction)
scale: 0.5  # image scale (+/- gain)
shear: 0.0  # image shear (+/- deg)
perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
flipud: 0.5  # image flip up-down (probability)
fliplr: 0.5  # image flip left-right (probability)
mosaic: 1.0  # image mosaic (probability)
mixup: 0.5  # image mixup (probability)
copy_paste: 0.0  # segment copy-paste (probability)
'''


# In[13]:


data = '''
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]

path: ../yolo_data/fold0/  # dataset root dir
train: images/train  # train images (relative to 'path') 128 images
val: images/val  # val images (relative to 'path') 128 images
test:  # test images (optional)

# Classes
nc: 1  # number of classes
names: ['reef']  # class names


# Download script/URL (optional)
# download: https://ultralytics.com/assets/coco128.zip
'''


# In[14]:


# !git clone https://github.com/ultralytics/yolov5.git
get_ipython().system('git clone https://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@github.com/soumya997/yolov5-w-f2-mod.git')
get_ipython().system('mv ./yolov5-w-f2-mod ./yolov5')


# In[15]:


with open('./yolov5/data/reef_f1_naive.yaml', 'w') as fp:
    fp.write(data)
with open('./yolov5/data/hyps/hyp.heavy.2.yaml', 'w') as fp:
    fp.write(hyps)


# In[16]:


get_ipython().run_line_magic('cd', 'yolov5')


# In[17]:


get_ipython().system('ls data/')


# In[18]:


get_ipython().system('python -m wandb disabled')

get_ipython().system('python train.py     --img 3584     --batch 2     --epochs 11     --data data/reef_f1_naive.yaml     --weights yolov5s6.pt     --name base_vid_2val     --hyp data/hyps/hyp.heavy.2.yaml     --save-period 1')


# In[19]:


# !python -m wandb disabled

# !python train.py \
#     --img 3000 \
#     --batch 2 \
#     --epochs 11 \
#     --data data/reef_f1_naive.yaml \
#     --weights yolov5s6.pt \
#     --name cots_with_albs \
#     --hyp data/hyps/hyp.heavy.2.yaml \
#     --save-period 1


# In[20]:


get_ipython().system('ls')


# In[21]:


get_ipython().run_line_magic('cd', '/kaggle/working')

get_ipython().system('cp -r /kaggle/working/yolov5/runs/train/base_vid_2val /kaggle/working')

get_ipython().system('cp /kaggle/working/yolov5/data/reef_f1_naive.yaml /kaggle/working/base_vid_2val/')
get_ipython().system('cp /kaggle/working/yolov5/data/hyps/hyp.heavy.2.yaml /kaggle/working/base_vid_2val/')
# !cp /kaggle/working/yolov5/utils/augmentations.py /kaggle/working/base_vid_2val/

get_ipython().system('rm -r /kaggle/working/yolov5')
get_ipython().system('rm -r /kaggle/working/images')
get_ipython().system('rm -r /kaggle/working/labels')

