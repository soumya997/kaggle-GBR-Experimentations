#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from shutil import copyfile
from tqdm import tqdm


# In[2]:


train = pd.read_csv('../input/tensorflow-great-barrier-reef/train.csv')
train['pos'] = train.annotations != '[]'


# In[3]:


train.head()


# In[4]:


get_ipython().system('mkdir -p ./yolo_data/fold1/images/val')
get_ipython().system('mkdir -p ./yolo_data/fold1/images/train')

get_ipython().system('mkdir -p ./yolo_data/fold1/labels/val')
get_ipython().system('mkdir -p ./yolo_data/fold1/labels/train')


# In[5]:


fold = 1

annos = []
for i, x in tqdm(train.iterrows()):
#     print(i,x)
#     break
    if x.video_id == fold:
        mode = 'val'
    else:
        # train
        mode = 'train'
        if not x.pos: continue
        # val
    copyfile(f'../input/tensorflow-great-barrier-reef/train_images/video_{x.video_id}/{x.video_frame}.jpg',
                f'./yolo_data/fold{fold}/images/{mode}/{x.image_id}.jpg')
    if not x.pos:
        continue
    r = ''
    anno = eval(x.annotations)
    for an in anno:
        r += '0 {} {} {} {}\n'.format((an['x'] + an['width'] / 2) / 1280,
                                        (an['y'] + an['height'] / 2) / 720,
                                        an['width'] / 1280, an['height'] / 720)
#         print(r)
#         print()
    with open(f'./yolo_data/fold{fold}/labels/{mode}/{x.image_id}.txt', 'w') as fp:
        fp.write(r)


# In[ ]:





# In[6]:


train["video_id"].value_counts()


# In[7]:


import os
len(os.listdir("./yolo_data/fold1/images/train"))


# In[8]:


hyps = '''
# YOLOv5 Hyperparameter Evolution Results
# Best generation: 4
# Last generation: 5
#    metrics/precision,       metrics/recall,      metrics/mAP_0.5, metrics/mAP_0.5:0.95,         val/box_loss,         val/obj_loss,         val/cls_loss
#              0.16318,              0.11858,             0.064654,             0.023477,             0.039919,            0.0055961,                    0

lr0: 0.01048
lrf: 0.11894
momentum: 0.92909
weight_decay: 0.00052
warmup_epochs: 3.88107
warmup_momentum: 0.57065
warmup_bias_lr: 0.10278
box: 0.06726
cls: 0.46752
cls_pw: 1.44082
obj: 1.0107
obj_pw: 0.803
iou_t: 0.2
anchor_t: 3.1544
fl_gamma: 0.0
hsv_h: 0.01506
hsv_s: 0.79959
hsv_v: 0.49056
degrees: 0.0
translate: 0.09677
scale: 0.48172
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 0.95352
mixup: 0.0
copy_paste: 0.0
anchors: 2.55656
'''


# In[9]:


data = '''
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../yolo_data/fold1/  # dataset root dir
train: images/train  # train images (relative to 'path') 128 images
val: images/val  # val images (relative to 'path') 128 images
test:  # test images (optional)

# Classes
nc: 1  # number of classes
names: ['reef']  # class names


# Download script/URL (optional)
# download: https://ultralytics.com/assets/coco128.zip
'''


# In[10]:


get_ipython().system('git clone https://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@github.com/soumya997/yolov5-w-f2-mod.git')


# In[11]:


get_ipython().system('mv ./yolov5-w-f2-mod ./yolov5')


# In[12]:


# !touch ./data/reef_f1_naive.yaml
# !touch ./data/hyps/hyp.heavy.2.yaml


# In[13]:


# !pwd


# In[14]:


with open('./yolov5/data/reef_f1_naive.yaml', 'w') as fp:
    fp.write(data)
with open('./yolov5/data/hyps/hyp.heavy.2.yaml', 'w') as fp:
    fp.write(hyps)


# In[15]:


get_ipython().run_line_magic('cd', 'yolov5')


# In[16]:


get_ipython().system('ls data')


# In[17]:


get_ipython().system('python train.py --img 1920 --batch 2 --epochs 7 --data reef_f1_naive.yaml --weights yolov5s6.pt --name l6_3600_uflip_vm5_f1 --hyp data/hyps/hyp.heavy.2.yaml')


# In[18]:


# !python train.py --img 1920\ 
# --batch 2\ 
# --epochs 7\ 
# --data reef_f1_naive.yaml\ 
# --weights yolov5s6.pt\
# --name base_yolo_5\ 
# --hyp data/hyps/hyp.heavy.2.yaml


# In[19]:


# asdlkasnhdljnalsdnlaksndlkasnd hvkvk khvhkv  apijdpajsd apsjdpa asnpduhy7hy7hyn asdnpa apsjdpa asnpdn asdnpa iygiyugb giugu


# # Kaggle dataset API call:

# In[20]:


get_ipython().system('rm /root/.kaggle')


# In[21]:


get_ipython().system('mkdir /root/.kaggle')


# In[22]:


get_ipython().system('cp -v /kaggle/input/random-private-files/kaggle.json /root/.kaggle')


# In[23]:


get_ipython().system('kaggle datasets init -p ./runs/train/base_yolo_5/weights')


# In[24]:


get_ipython().run_cell_magic('writefile', './runs/train/base_yolo_5/weights/dataset-metadata.json', '\n{\n  "title": "yolov5_train_2_weights",\n  "id": "soumya9977/yolov5-train-2-weights",\n  "licenses": [\n    {\n      "name": "CC0-1.0"\n    }\n  ]\n}')


# In[ ]:


get_ipython().system('kaggle datasets create -p ./runs/train/base_yolo_5/weights')


# In[ ]:


# !kaggle datasets version -p ./runs/train/l6_3600_uflip_vm5_f1/weights -m "adding model weights"


# In[ ]:


# !cp -v ./runs/train/l6_3600_uflip_vm5_f1/dataset-metadata.json ./runs/train/l6_3600_uflip_vm5_f1/weights


# In[ ]:


get_ipython().system('ls -GFlash --color .')


# # copying to the working:

# In[ ]:


get_ipython().system('cp -R ./runs/train/base_yolo_5 /kaggle/working')


# In[ ]:




