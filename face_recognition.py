# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 19:54:46 2021

@author: Julia
"""


import torch
import numpy as np
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


#imshow -- делаем картинку обратно из pytorch тензора
def imshow(img):
    plt.imshow(np.transpose(img, (1, 2, 0)))


#задаем ту же модель, что и при обучении
import cv2 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 16, 5)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32*53*53, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 2)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 32 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.softmax(self.fc3(x))
        return x

model = Net()
#загружаем веса
model.load_state_dict(torch.load('model-1.pt'))
model.eval()


from PIL import Image

import cv2 as cv

#Вырезаем из группового фото отдельные лица с помощью pretrained Haar feature-based cascade classifiers
original_image = cv.imread('face_recogn_test.jpg')
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')
detected_faces = face_cascade.detectMultiScale(grayscale_image)

#это костыль, у всех изображений в датасете была прямоугольная форма, притом одинаковая, а Haar classifier вырезает квадрат
loader = transforms.Compose([transforms.Resize(size=(224,224)),
     transforms.CenterCrop([220,200]),transforms.Resize(size=(224,224))])
    
    
loader2 =  transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])   
for (column, row, width, height) in detected_faces:
    #вырезали лицо с группового фото
    crop_img = original_image[int(row):int((row+height)),int(column):int((column+width))]
    img = Image.fromarray(np.array(crop_img))
    #делаем из картинки тензор torch
    image = loader2(loader(img)).float()
    imshow(image)
    plt.show()
    #модель на вход принимает только бэтчи, отдельные изображения не принимаются, поэтому добавляем еще одно измерение
    image = image.unsqueeze(0)
    
    output = model(image)   
    _, pred = torch.max(output, 1) 
    if (pred == 1):
        print('female')
    if (pred == 0):
        print('male')
        
    


