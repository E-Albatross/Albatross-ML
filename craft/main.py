import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import cv2

import craft_utils

from craft import CRAFT

## parameters
craft_model = 'craft_mlt_25k.pth' # pretrained model
text_threshold = 0.7 # text confidence threshold
low_text = 0.4
link_threshold = 0.5
cuda = False # use cuda for inference
mag_ratio = 1.0
border = 5

font_images = []
user_images = []
syllables = [] # 각 줄에 대한 음절들

## 이미지 불러오기
img_path = 'test.png'

original_img = cv2.imread(img_path)
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
canvas_size = original_img.shape[1]

## 줄 별로 텍스트 자르기
height = original_img.shape[0] // 12
for i in range(0,10,3):
    y = height * i
    fnt = original_img[height*i:height*(i+1),:].copy()
    usr = original_img[height*(i+2):height*(i+3), :].copy()

    font_images.append(fnt)
    user_images.append(usr)

## 음절 단위 추출
# load net - CRAFT
net = CRAFT()     # initialize
print('Loading weights from checkpoint (' + craft_model + ')')
if cuda:
    net.load_state_dict(craft_utils.copyStateDict(torch.load(craft_model)))
else:
    net.load_state_dict(craft_utils.copyStateDict(torch.load(craft_model, map_location='cpu')))

if cuda:
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False

net.eval()

# load data
image_list = user_images
for k, image in enumerate(image_list):
    bbox= craft_utils.test_net(net, image, canvas_size, text_threshold,low_text, mag_ratio,cuda)

    img_ = image.copy()
    cropped_img = []
    color = [(0,255,0), (0,0,255)] # bbox 확인용
    for i in range(len(bbox)-1):
        x, y, w, h = bbox[i]
        x_next = bbox[i+1][0]
        if (x+w) > x_next: err = (x+w - x_next) // 2
        w -= err
        cv2.rectangle(img_, (x,y), (x+w, y+h), color[i%2], 2) # bbox 확인용
        crop = image[border:, x:x+w].copy()
        cropped_img.append(crop)

    # 마지막 인덱스
    x, y, w, h = bbox[-1]
    cv2.rectangle(img_, (x, y), (x+w, y + h), color[i % 2], 2)
    crop = image[:, x:x+w].copy()
    cropped_img.append(crop)

    syllables.append(cropped_img)
    cv2.imshow('img',img_)
    cv2.waitKey(0)

    # 이미지 확인용
    for img in cropped_img:
        cv2.imshow('img',img)
        cv2.waitKey(0)