import numpy as np
import cv2
import math

import torch
from torch.autograd import Variable

import imgproc

from collections import OrderedDict

## 모델 불러오기
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def getDetBoxes(textmap, text_threshold, low_text):
    # prepare data
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score.astype(np.uint8), connectivity=4)
    det = []

    for k in range(1,nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue

        # thresholding
        if np.max(textmap[labels==k]) < text_threshold: continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1

        # boundary check
        if sx < 0 : sx = 0
        if sy < 0 : sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
        x, y, w, h = cv2.boundingRect(np_contours)
        box = [x, y, w, h]

        det.append(box)
    det.sort(key=lambda x: x[0])

    return det

def adjustResultCoordinates(boxes, ratio_w, ratio_h, ratio_net = 2):
    boxes = np.array(boxes, dtype=np.float32)

    for box in boxes:
        box[0] *= ratio_w * ratio_net
        box[1] *= ratio_h * ratio_net
        box[2] *= ratio_w * ratio_net
        box[3] *= ratio_h * ratio_net

    return boxes


def test_net(net, image, canvas_size, text_threshold, low_text, mag_ratio, cuda):
    # resize
    img_resized, target_ratio = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]

    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score map
    score_text = y[0,:,:,0].cpu().data.numpy()

    boxes= getDetBoxes(score_text, text_threshold, low_text)
    boxes = np.array(boxes, dtype=np.int32)
    boxes *= 2 # ratio_net = 2

    # boxes= adjustResultCoordinates(boxes, ratio_w, ratio_h).astype(np.int32) # 비율 변경

    return boxes