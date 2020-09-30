#!/usr/bin/env python3
''' Written by jared.vasquez@yale.edu '''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

def loadmodel():
    PATH = './models/model_test14_FINAL.pt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # criterion = nn.CrossEntropyLoss()
    # model = CNN()
    # model.load_state_dict(torch.load(PATH, map_location=device))
    model = torch.load(PATH)
    model.eval()
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
            img = np.float32(roi)/255.
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=-1)
            # print(img.shape)
            transpose_img = np.transpose(img, (0, 3, 2, 1))
            # print(transpose_img.shape)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            inp = torch.from_numpy(transpose_img)
            out = model(inp)
            _, prediction = torch.max(out, 1)
            # print(prediction)
            pred = classes[prediction]
            cv2.putText(window, 'Prediction: %s' % (pred), (fx,fy+2*fh), font, 1.0, (245,210,65), 2, 1)
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
        #input_channels = 1, output channels = 6, kernel_size=(3,3)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2) # same as pool_size=(2,2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        self.pool4 = nn.MaxPool2d(2)

        self.hidden= nn.Linear(128*16*16, 512) # The linear layer should therefore take 128*16*16=32768 input features
        self.drop = nn.Dropout(0.3) # 30% probability of an element to be zeroed
        self.out = nn.Linear(512, 6)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # -> [batch_size, 32, 298, 298] => The 32 is given by the number of kernels your conv layer is using.
                                                                    # Since you are not using any padding and leave the stride and dilation as 1,
                                                                    # a kernel size of 3 will crop 1 pixel in each spatial dimension.
                                                                    # Therefore you’ll end up with 32 activation maps of spatial size 298x298.
        # print(x.shape)
        x = self.pool1(x) # -> [batch_size, 32, 149, 149] => The max pooling layer will halve the spatial size
        # print(x.shape)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = x.flatten(start_dim=1)
        #x = torch.flatten(x, start_dim=1) # Flattens a contiguous range of dims in a tensor
        #x = x.view(x.size(0), -1) # -> [batch_size, 128*16*16=32768]
        x = F.relu(self.hidden(x)) # -> [batch_size, 512]
        x = self.drop(x) # -> [batch_size, 512]
        x = self.out(x)
        x = F.softmax(x, dim=1) # -> [batch_size, 6]
        return x


if __name__ == '__main__':
    initClass('NONE')
    main()
