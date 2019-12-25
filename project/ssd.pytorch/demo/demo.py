from __future__ import print_function

import torch
from torch.autograd import Variable
import cv2
import time
from imutils.video import FPS, WebcamVideoStream
import argparse
import sys
import os
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
from data import config as global_config

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

def parse_args():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
    parser.add_argument('--weights', default='weights/ssd_300_VOC0712.pth',
                        type=str, help='Trained state_dict file path')
    parser.add_argument('--cuda', default=False, type=bool,
                        help='Use cuda in live demo')
    parser.add_argument('--img_dir', default=None,
                        type=str, help='test image dir',required=True)
    parser.add_argument('--video_dir', default=None,
                        type=str, help='test video dir')
    parser.add_argument('--wait_time', default=0,
                        type=int, help='cv2 waitkey time')
    args = parser.parse_args()
    return args

def predict(frame,net, transform):
    height, width = frame.shape[:2]
    x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    y = net(x)  # forward pass
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor([width, height, width, height])
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            cv2.rectangle(frame,
                          (int(pt[0]), int(pt[1])),
                          (int(pt[2]), int(pt[3])),
                          COLORS[i % 3], 2)
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1
    return frame

def video_demo(video_list, net, transform, wait_time = 0):
    for filename in video_list:
        stream = cv2.VideoCapture(filename)
        if not stream.isOpened():
            print("Failed to open ", filename)
            continue
        while True:
            frame = stream.read()
            if frame is None:
                break
            key = cv2.waitKey(wait_time) & 0xFF
            frame = predict(frame, net, transform)
            # keybindings for display
            if key == ord('p'):  # pause
                while True:
                    key2 = cv2.waitKey(1) or 0xff
                    cv2.imshow('frame', frame)
                    if key2 == ord('p'):  # resume
                        break
            cv2.imshow('frame', frame)
            if key == 27:  # exit
                return

def img_demo(img_list, net, transform, wait_time = 0):
    for filename in img_list:
        frame = cv2.imread(filename)
        if frame is None:
            continue
        key = cv2.waitKey(wait_time) & 0xFF
        frame = predict(frame, net, transform)
        # keybindings for display
        if key == ord('p'):  # pause
            while True:
                key2 = cv2.waitKey(1) or 0xff
                cv2.imshow('frame', frame)
                if key2 == ord('p'):  # resume
                    break
        cv2.imshow('frame', frame)
        if key == 27:  # exit
            break


if __name__ == '__main__':

    args = parse_args()
    wait_time = args.wait_time
    net = build_ssd('test', 300, 21, batch_norm = global_config.BATCH_NORM)    # initialize SSD
    net.load_state_dict(torch.load(args.weights))
    transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))
    net = net.eval()

    if args.img_dir is not None:
        img_list = []
        for dirpath, dirnames, filenames in os.walk(args.img_dir):
            for filename in filenames:
                if filename.endswith(('.jpg','.png','bmp')):
                    img_list.append(os.path.join(dirpath,filename))
        img_demo(img_list, net, transform, wait_time)
    elif args.video_dir is not None:
        video_list = []
        for dirpath, dirnames, filenames in os.walk(args.img_dir):
            for filename in filenames:
                if filename.endswith(('.mkv','.mp4','3gp', '.mkv')):
                    video_list.append(os.path.join(dirpath,filename))
        video_demo(video_list, net, transform, wait_time)




