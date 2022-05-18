# import logging

import bentoml

import numpy as np
import torch
from torchvision import transforms
import cv2
import os

from bentoml.io import Image, JSON, Text
from PIL.Image import Image as PILImage

import utils

#
img_size = (224, 224)

# Load the runner
craft_runner = bentoml.pytorch.load_runner(
    "craft:latest",
)
# fpn_runner = bentoml.pytorch.load_runner(
#     "fpn:latest"
# )
# seg_runner = bentoml.pytorch.load_runner(
#     "segmentation:latest"
# )

svc = bentoml.Service(
    name="brgs_ai",
    runners=[
        craft_runner#,  seg_runner
    ],
)



trans = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
])
# root_dir = './output/fntout'
# out_dir = os.path.join(root_dir, 'bad1')
# cout_dir = os.path.join(out_dir, 'line1')
#
# if not os.path.exists(out_dir):
#     os.mkdir(out_dir)
# if not os.path.exists(cout_dir):
#     os.mkdir(cout_dir)

boundingboxes = {}


@svc.api(input=Image(), output=JSON())
async def predict(f: PILImage) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
    assert isinstance(f, PILImage)
    file={}
    # preprocessing
    image = np.array(f.convert('RGB'))
    height = image.shape[0] // 12
    border = 7
    images=[]   # 줄 단위 이미지
    for i in range(0, 10, 3):
        usr = image[height * (i+2) + border:height * (i + 3) - border, :].copy()
        images.append(usr)

    # 음절 분리
    syllable_boxes = {}
    character_boxes = {}
    num=1
    for k, img in enumerate(images):
        img_ = img.copy()       # bbox 확인용
        image = utils.imgproc(img)      # resize image and nomalization
        x = torch.from_numpy(image).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        y = await craft_runner.async_run(x)
        score_text = y[0,:,:,0].cpu().data.numpy()
        det = utils.getDetBoxes(score_text)

        cropped_img = []
        bbox = []
        color = [(0, 255, 0), (0, 0, 255)]  # bbox 확인용
        for i in range(len(det)-1):
            x, y, w, h, cy = det[i]
            x_next = det[i + 1][0]
            if (x + w) > x_next: w -= (x + w - x_next) // 2
            det[i][2] = w
            #
            size = max(w,height-2*border)
            crop, b = utils.crop_img(img, size, width=w, height=height-2*border, x=x)
            bbox.append(b)
            cropped_img.append(crop)
        x, y, w, h, cy = det[-1]
        size = max(w, height-2*border)
        crop, b = utils.crop_img(img, size, width=w, height=height-2*border, x=x)
        cropped_img.append(crop)
        bbox.append(b)

        # draw rectangle
        for i, box in enumerate(bbox):
            x,y,w,h,cy = box
            cv2.rectangle(img_, (x, y), (x + w, y + h), color[i%2], 2)  # bbox 확인용

        syllables_img = np.array(cropped_img)

        syllable_boxes[k] = bbox

        # 확인용
        # cv2.imwrite(os.path.join(out_dir, f'bad1_{k}.png'), img_)
        #
        # # 음소 분리
        # syllables = np.where(syllables_img < 170, 1, 0).astype(np.float32) # threshold = 200
        # character_in_line = []
        # for idx, syllable in enumerate(syllables):
        #     img_ = syllables_img[idx]
        #     # print(f'syllable: {num}')
        #     x = trans(syllable)
        #     # y = await fpn_runner.async_run(x)
        #     y = await seg_runner.async_run(x)
        #     y = torch.sigmoid(y)
        #     seg = y.data.numpy()
        #     seg_result = utils.masks_to_colorimg(seg) # 확인용
        #     bbox = utils.getDetBoxes_from_seg(syllable, seg)
        #     colors = [(255,0,0),(0,0,255),(0,255,0)]
        #     character_in_line.append(bbox)
        #     # 보정 후 - concat
        #     syllable_ = syllable.copy()*255
        #     for i, box in enumerate(bbox):
        #         x, y, w, h= box
        #         cv2.rectangle(syllable_, (x, y), (x + w, y + h), colors[i], 4)  # bbox 확인용

            # # 보정 후 - not concat
            # for i, bboxes in enumerate(bbox):
            #     for box in bboxes:
            #         if box == -1:
            #             continue
            #         x, y, w, h= box
            #         cv2.rectangle(seg_result, (x, y), (x + w, y + h), colors[i], 2)  # bbox 확인용

            # # 보정 전
            # for i, ch in enumerate(bbox):
            #     for box in ch:
            #         print(f'box {box}')
            #         if not box['label']==-1:
            #             x, y, w, h = box['box']
            #  # 음소 분리
        #         # syllables = np.where(syllables_img < 170, 1, 0).astype(np.float32) # threshold = 200
        #         # character_in_line = []
        #         # for idx, syllable in enumerate(syllables):
        #         #     img_ = syllables_img[idx]
        #         #     # print(f'syllable: {num}')
        #         #     x = trans(syllable)
        #         #     # y = await fpn_runner.async_run(x)
        #         #     y = await seg_runner.async_run(x)
        #         #     y = torch.sigmoid(y)
        #         #     seg = y.data.numpy()
        #         #     seg_result = utils.masks_to_colorimg(seg) # 확인용
        #         #     bbox = utils.getDetBoxes_from_seg(syllable, seg)
        #         #     colors = [(255,0,0),(0,0,255),(0,255,0)]
        #         #     character_in_line.append(bbox)
        #         #     # 보정 후 - concat
        #         #     syllable_ = syllable.copy()*255
        #         #     for i, box in enumerate(bbox):
        #         #         x, y, w, h= box
        #         #         cv2.rectangle(syllable_, (x, y), (x + w, y + h), colors[i], 4)  # bbox 확인용           cv2.rectangle(seg_result, (x, y), (x + w, y + h), colors[i], 2)  # bbox 확인용
            #
            # cv2.imwrite(os.path.join(cout_dir,f'{str(num)}.png'),syllable_)
            # num+=1
        # character_boxes[k] = character_in_line

    file['syllable']=syllable_boxes
    # file['character']=character_boxes

    return file

