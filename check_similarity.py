#!/usr/bin/env python
# coding: utf-8

# # Load Image and Run Similarity Check

# In[6]:


import torch
import torch.nn as nn
from torchvision.models import resnet18
from collections import namedtuple
import matplotlib.pyplot as plt

from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from torchvision import transforms, datasets
import numpy as np
import torchvision.transforms.functional as F

import sys
from sklearn.metrics.pairwise import cosine_similarity


preprocess = transforms.Compose([
    transforms.Resize(256), #256
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#create DL architecture from resnet18
class Resnet18(torch.nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        features = list( resnet18(pretrained = True).children() )[:8]
        self.features = nn.ModuleList(features).eval() 
        
    def forward(self, x):
        with torch.no_grad():
            layer_outputs = []
            for layer,model in enumerate(self.features):
                x = model(x)
            avg = nn.AdaptiveAvgPool2d( (1, 1) ) #average the last layer
            feature = avg(x).squeeze()

            
            return feature.flatten() # return the image feature


# In[7]:


# filename from the terminal
file_name = sys.argv[1]



# In[8]:


# Load the feature vector of 8 given apple images
embedding = np.loadtxt(open("embedding.csv", "r"), delimiter=",")


# In[11]:


# find the feature vector of the new image
test_m = Resnet18()
try:
    rgba_image = Image.open(f'test/{file_name}')
    input_image = rgba_image.convert('RGB')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model by insering a dimension to the model

    embb = test_m.forward( input_batch )
    embb = np.array(embb)
except:
    print('Some Error in processing')


# In[ ]:


#loop over all the embedding and compute similarity
#take vote of majority i.e if similarity value is more than 0.7 for more than 5 images then image is similar
vote = 0 # take vote for each image on similarity
for em in embedding:
    cos_sim = cosine_similarity(em.reshape(1, -1), embb.reshape(1, -1))
    if cos_sim >= 0.7:
        vote += 1


# ## Generate Dialog Box

# In[ ]:

# ----- Tkinter package is used to display the message
# import tkinter as tk
# from tkinter import messagebox
# root = tk.Tk()
# root.withdraw()

if vote > 4: # 5 images agreeing that new image is apple
    # messagebox.showwarning('Information', 'yay! Apple')
    print('Information: yay! an APPLE')
else:
#     messagebox.showwarning('Information', 'Nope')
    print('Information: oops! NOT an APPLE')

