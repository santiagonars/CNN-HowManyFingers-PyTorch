#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import copy
import cv2
import os


dataColor = (0,255,0)
font = cv2.FONT_HERSHEY_SIMPLEX
fx, fy, fh = 10, 50, 45
takingData = 0
className = 'NONE'
count = 0
showMask = 0


classes = 'FIVE FOUR NONE ONE THREE TWO'.split()


def initClass(name):
    global className, count
    className = name
    os.system('mkdir -p data/%s' % name)
    count = len(os.listdir('data/%s' % name))


def binaryMask(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (7,7), 3)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, new = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return new


def preProcessImage(roi):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(300),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])
    img_t = transform(roi)
    batch_t = torch.unsqueeze(img_t, 0)
    return batch_t
    
    
def loadmodel():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # seems to work with this line
    # Load state_dict: (Recommended)
    PATH = './models/model_test14.pth'
    model = CNN()
    if device == torch.device('cpu'): # load on CPU
        model.load_state_dict(torch.load(PATH, map_location=device)) # must deserialize the saved state_dict before passing
    else: # load on GPU
        model.load_state_dict(torch.load(PATH))
        model.to(device)
    model.eval() # set dropout and batch normalization layers to evaluation mode

    # Load Entire Model:
    # NOTE: was save the entire module using Python’s pickle module; Disadvantage: serialized data is bound to the specific classes
    #                                                                and the exact directory structure used when the model is saved
    """ PATH = './models/model_test14_entireModel.pt'
    model = torch.load(PATH)
    model.eval() """
    return model


def main():
    global font, size, fx, fy, fh
    global takingData, dataColor
    global className, count
    global showMask

    model = loadmodel()

    x0, y0, width = 200, 220, 300

    cam = cv2.VideoCapture(0)
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)

    while True:
        # Get camera frame
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1) # mirror
        window = copy.deepcopy(frame)
        cv2.rectangle(window, (x0,y0), (x0+width-1,y0+width-1), dataColor, 12)

        # draw text
        if takingData:
            dataColor = (0,250,0)
            cv2.putText(window, 'Data Taking: ON', (fx,fy), font, 1.2, dataColor, 2, 1)
        else:
            dataColor = (0,0,250)
            cv2.putText(window, 'Data Taking: OFF', (fx,fy), font, 1.2, dataColor, 2, 1)
        cv2.putText(window, 'Class Name: %s (%d)' % (className, count), (fx,fy+fh), font, 1.0, (245,210,65), 2, 1)

        # get region of interest
        roi = frame[y0:y0+width,x0:x0+width]
        roi = binaryMask(roi)

        # apply processed roi in frame
        if showMask:
            window[y0:y0+width,x0:x0+width] = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        # take data or apply predictions on ROI
        if takingData:
            cv2.imwrite('data/{0}/{0}_{1}.png'.format(className, count), roi)
            count += 1
        else:
            img = preProcessImage(roi)
            out = model(img)
            # find prediction with the maximum score in output vector 'out'
            _, pred = torch.max(out, 1) 
            prediction = classes[pred[0]]
            cv2.putText(window, 'Prediction: %s' % (prediction), (fx,fy+2*fh), font, 1.0, (245,210,65), 2, 1)
            # use below for demoing purposes
            #cv2.putText(window, 'Prediction: %s' % (pred), (x0,y0-25), font, 1.0, (255,0,0), 2, 1)

        # show the window
        cv2.imshow('Original', window)

        # Keyboard inputs
        key = cv2.waitKey(10) & 0xff

        # use q key to close the program
        if key == ord('q'):
            break

        # Toggle data taking
        elif key == ord('s'):
            takingData = not takingData

        elif key == ord('b'):
            showMask = not showMask

        # Toggle class
        elif key == ord('0'):  initClass('NONE')
        elif key == ord('`'):  initClass('NONE') # because 0 is on other side of keyboard
        elif key == ord('1'):  initClass('ONE')
        elif key == ord('2'):  initClass('TWO')
        elif key == ord('3'):  initClass('THREE')
        elif key == ord('4'):  initClass('FOUR')
        elif key == ord('5'):  initClass('FIVE')

        # adjust the size of window
        #elif key == ord('z'):
        #    width = width - 5
        #elif key == ord('a'):
        #    width = width + 5

        # adjust the position of window
        elif key == ord('i'):
            y0 = max((y0 - 5, 0))
        elif key == ord('k'):
            y0 = min((y0 + 5, window.shape[0]-width))
        elif key == ord('j'):
            x0 = max((x0 - 5, 0))
        elif key == ord('l'):
            x0 = min((x0 + 5, window.shape[1]-width))

    cam.release()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        self.pool4 = nn.MaxPool2d(2)

        self.hidden= nn.Linear(128*16*16, 512)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(512, 6)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = x.flatten(start_dim=1)
        x = F.relu(self.hidden(x)) 
        x = self.drop(x)
        x = self.out(x)
        x = F.softmax(x, dim=1)
        return x


if __name__ == '__main__':
    initClass('NONE')
    main()
