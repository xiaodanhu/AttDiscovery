from data import *
import pandas as pd
import matplotlib.pyplot as plt
import glob
import torch
import clip
from PIL import Image
import numpy as np
import os, json, cv2
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from model import *
from torch.optim.lr_scheduler import LambdaLR, StepLR
import random
from transformers import CLIPProcessor, CLIPVisionModel
from torch.utils.data import DataLoader, Subset, random_split

clip = torch.jit.load("../checkpoints/model.pt").cuda().eval()
device = "cuda"
data_root = '/media/hxd/82231ee6-d2b3-4b78-b3b4-69033720d8a8/MyDatasets/amazon'

random.seed(77)
error = nn.CrossEntropyLoss().cuda()

############ First task: color ############
cur_attr = 'color'
PATH_color = "./checkpoints/color.pth"
version = 'img_by_attr_50'

le_color = preprocessing.LabelEncoder()
le_color.fit(next(os.walk(data_root + '/' + version + '/' + cur_attr))[1])

attr_img_paths, attrs_labels = load_data(cur_attr)

imgs = []
for img_path in attr_img_paths:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img) # convert cv2 -> PIL image
    imgs.append(img)

model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
inputs = processor(images=imgs, return_tensors="pt")
with torch.no_grad():
    vision_outputs = model.vision_model(**inputs)
image_embeds = vision_outputs[1]

dataset = Dataset(image_embeds, le_color, attrs_labels, list(range(len(image_embeds))))

# generate subset based on indices
color_test_indices = random.sample(dataset.data_indx, int(len(image_embeds)*0.2))
color_train_set = Subset(dataset, [i for i in dataset.data_indx if i not in color_test_indices])
color_test_set = Subset(dataset, color_test_indices)
color_trainloader = DataLoader(color_train_set, batch_size=64, shuffle=True)
color_testloader = DataLoader(color_test_set, batch_size=64, shuffle=True)

# torch.save({'model_encode': net.encode.state_dict(), 'model_color': net.color.state_dict()}, PATH_color)
net = Net(len(le_color.classes_), 4).cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
scheduler_color = StepLR(optimizer, step_size=10, gamma=0.1)

num_epochs = 301

epoch_list = []
train_accuracy_list_color = []; train_accuracy_list_item_shape = []
train_loss_list_color = []; train_loss_list_item_shape = []
valid_accuracy_list_color = []; valid_accuracy_list_item_shape = []

for epoch in range(num_epochs):
    correct_color = 0
    running_loss_color = 0
    cnt_color = 0
    net.train()

    for _, data in enumerate(color_trainloader):

        optimizer.zero_grad()
        
        image_batch, label_batch = data['image'], data['label']
        image_batch = image_batch.cuda(); label_batch = label_batch.cuda()

        outputs = net(image_batch)[0]
        loss = error(outputs, label_batch)
        loss.backward()
        
        optimizer.step()
        predictions_color = torch.max(outputs, 1)[1].cuda()
        correct_color += (predictions_color == label_batch).sum()
        running_loss_color += loss.item()
        cnt_color += len(label_batch)
    
    train_loss_list_color.append(float(running_loss_color) / float(len(color_trainloader)))
    train_accuracy_list_color.append(float(correct_color) / cnt_color)
    
    # test on validation set
    correct_color = 0
    cnt_color = 0
    with torch.no_grad():
        for _, data in enumerate(color_testloader):

            image_batch, label_batch = data['image'], data['label']
            image_batch = image_batch.cuda(); label_batch = label_batch.cuda()

            outputs = net(image_batch)[0]
            
            predictions_color = torch.max(outputs, 1)[1].cuda()
            correct_color += (predictions_color == label_batch).sum()
            cnt_color += len(label_batch)

        valid_accuracy_list_color.append(float(correct_color) / cnt_color)
    
    if (epoch % 20) == 0:
        print("Color: Epoch: {}, train_loss: {}, train_accuracy: {}%, test_accuracy: {}%".format(epoch, 
                                            train_loss_list_color[-1], 
                                            train_accuracy_list_color[-1]*100, 
                                            valid_accuracy_list_color[-1]*100))
        torch.save({
            'model_encode': net.encode.state_dict(), 
            'model_color': net.color.state_dict()
            }, PATH_color)

'''
############ Second task: shape ############
'''
cur_attr = 'item_shape'
pre_attr = 'color'
PATH_color = "./checkpoints/color.pth"
PATH_item_shape = "./checkpoints/shape.pth"

le_color = preprocessing.LabelEncoder()
le_color.fit(next(os.walk(data_root + '/' + version + '/' + pre_attr))[1])
le_item_shape = preprocessing.LabelEncoder()
le_item_shape.fit(next(os.walk(data_root + '/' + version + '/' + cur_attr))[1])

pattr_img_paths, pattrs_labels = load_data(pre_attr, cur_attr)
cattr_img_paths, cattrs_labels = load_data(cur_attr)

imgs = []
for img_path in np.concatenate([pattr_img_paths, cattr_img_paths]):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img) # convert cv2 -> PIL image
    imgs.append(img)

model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
inputs = processor(images=imgs, return_tensors="pt")
with torch.no_grad():
    vision_outputs = model.vision_model(**inputs)
image_embeds = vision_outputs[1]

pattr_img_fea = image_embeds[:len(pattr_img_paths)]
cattr_img_fea = image_embeds[len(pattr_img_paths):]

cdataset = Dataset(cattr_img_fea, le_item_shape, cattrs_labels, list(range(len(cattr_img_fea))))
pdataset = Dataset(pattr_img_fea, le_color, pattrs_labels, list(range(len(pattr_img_fea))))

# generate subset based on indices
shape_test_indices = random.sample(cdataset.data_indx, int(len(cattr_img_fea)*0.2))
shape_train_set = Subset(cdataset, [i for i in cdataset.data_indx if i not in shape_test_indices])
shape_test_set = Subset(cdataset, shape_test_indices)
shape_trainloader = DataLoader(shape_train_set, batch_size=64, shuffle=True)
shape_testloader = DataLoader(shape_test_set, batch_size=64, shuffle=True)

# shape_train_set, shape_test_set = random_split(cdataset,[int(len(cattr_img_fea)*0.8),len(cattr_img_fea)-int(len(cattr_img_fea)*0.8)])
# shape_trainloader = DataLoader(shape_train_set, batch_size=64, shuffle=True)
# shape_testloader = DataLoader(shape_test_set, batch_size=64, shuffle=True)

color_train_set = Subset(pdataset, [i for i in pdataset.data_indx if i not in color_test_indices])
color_test_set = Subset(pdataset, color_test_indices)
color_trainloader = DataLoader(color_train_set, batch_size=64, shuffle=True)
color_testloader = DataLoader(color_test_set, batch_size=64, shuffle=True)

net = Net(len(le_color.classes_), len(le_item_shape.classes_)).cuda()
# torch.save({'model_encode': net.encode.state_dict(), 'model_color': net.color.state_dict()}, PATH_color)
color_state_dict = torch.load(PATH_color)
net.encode.load_state_dict(color_state_dict['model_encode'])
net.color.load_state_dict(color_state_dict['model_color'])


# coptimizer = torch.optim.Adam([
#             {'params': net.color.parameters()}
#             ], lr=1e-4)
# soptimizer = torch.optim.Adam([
#             {'params': net.encode.parameters()},
#             {'params': net.shape.parameters()}
#             ], lr=1e-2)
coptimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
soptimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
scheduler_color = StepLR(coptimizer, step_size=30, gamma=0.1)
scheduler_item_shape = StepLR(soptimizer, step_size=10, gamma=0.1)

num_epochs = 301

epoch_list = []
train_accuracy_list_color = []; train_accuracy_list_item_shape = []
train_loss_list_color = []; train_loss_list_item_shape = []
valid_accuracy_list_color = []; valid_accuracy_list_item_shape = []

for epoch in range(num_epochs):
    correct_color = 0; correct_item_shape = 0
    running_loss_color = 0; running_loss_item_shape = 0
    cnt_color = 0; cnt_item_shape = 0
    net.train()

    iterations = len(color_trainloader) * 2
    iterloader1 = iter(shape_trainloader)
    iterloader2 = iter(color_trainloader)

    for i in range(iterations):
        controller = i % 2 # controller is 0 for color, 1 for shape

        if controller == 0:
            data = next(iterloader2)
            coptimizer.zero_grad()
        else:
            try:
                data = next(iterloader1)
            except StopIteration:
                iterloader1 = iter(shape_trainloader)
                data = next(iterloader1)
            soptimizer.zero_grad()

        task0_mode = True if controller == 0 else False
        for name, param in net.named_parameters():
            if name.split('.')[0] == 'color':
                param.requires_grad = task0_mode
            elif name.split('.')[0] == 'shape':
                param.requires_grad = not task0_mode

        image_batch, label_batch = data['image'], data['label']
        image_batch = image_batch.cuda(); label_batch = label_batch.cuda()

        outputs = net(image_batch)[controller]
        loss = error(outputs, label_batch)
        loss.backward()
        
        if controller == 0:
            coptimizer.step()
            predictions_color = torch.max(outputs, 1)[1].cuda()
            correct_color += (predictions_color == label_batch).sum()
            running_loss_color += loss.item()
            cnt_color += len(label_batch)
        else:
            soptimizer.step()
            predictions_item_shape = torch.max(outputs, 1)[1].cuda()
            correct_item_shape += (predictions_item_shape == label_batch).sum()
            running_loss_item_shape += loss.item()
            cnt_item_shape += len(label_batch)
    
    train_loss_list_color.append(float(running_loss_color) / float(len(color_trainloader)))
    train_loss_list_item_shape.append(float(running_loss_item_shape) / float(len(color_trainloader)))
    train_accuracy_list_color.append(float(correct_color) / cnt_color)
    train_accuracy_list_item_shape.append(float(correct_item_shape) / cnt_item_shape)
    
    # test on validation set
    correct_color = 0; correct_item_shape = 0
    cnt_color = 0; cnt_item_shape = 0
    with torch.no_grad():
        iterations = len(color_testloader) * 2
        iterloader1 = iter(shape_testloader)
        iterloader2 = iter(color_testloader)

        for i in range(iterations):
            controller = i % 2 # controller is 0 for color, 1 for shape

            if controller == 0:
                data = next(iterloader2)
            else:
                try:
                    data = next(iterloader1)
                except StopIteration:
                    iterloader1 = iter(shape_testloader)
                    data = next(iterloader1)

            image_batch, label_batch = data['image'], data['label']
            image_batch = image_batch.cuda(); label_batch = label_batch.cuda()

            outputs = net(image_batch)[controller]
            
            if controller == 0:
                predictions_color = torch.max(outputs, 1)[1].cuda()
                correct_color += (predictions_color == label_batch).sum()
                cnt_color += len(label_batch)
            else:
                predictions_item_shape = torch.max(outputs, 1)[1].cuda()
                correct_item_shape += (predictions_item_shape == label_batch).sum()
                cnt_item_shape += len(label_batch)

        valid_accuracy_list_color.append(float(correct_color) / cnt_color)
        valid_accuracy_list_item_shape.append(float(correct_item_shape) / cnt_item_shape)
    

    
    if (epoch % 20) == 0:
        print("Color: Epoch: {}, train_loss: {}, train_accuracy: {}%, test_accuracy: {}%".format(epoch, 
                                            train_loss_list_color[-1], 
                                            train_accuracy_list_color[-1]*100, 
                                            valid_accuracy_list_color[-1]*100))
        print("Shape: Epoch: {}, train_loss: {}, train_accuracy: {}%, test_accuracy: {}%".format(epoch, 
                                            train_loss_list_item_shape[-1], 
                                            train_accuracy_list_item_shape[-1]*100, 
                                            valid_accuracy_list_item_shape[-1]*100))
        torch.save({
            'model_encode': net.encode.state_dict(), 
            'model_color': net.color.state_dict()
            }, PATH_color[:-4]+'2.pth')
        torch.save({
            'model_encode': net.encode.state_dict(), 
            'model_shape': net.shape.state_dict()
            }, PATH_item_shape[:-4]+'2.pth')

