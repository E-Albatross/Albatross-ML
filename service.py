import logging
import time

import bentoml

import numpy as np
import torch

from bentoml.io import Image, JSON
from PIL.Image import Image as PILImage

import utils

# 로그 생성
logger = logging.getLogger()

# 로그의 출력 기준 설정
logger.setLevel(logging.INFO)

# log 출력 형식
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# log 출력
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


#
craft_runner = bentoml.pytorch.load_runner(
    "craft:latest",
)

svc = bentoml.Service(
    name="craft",
    runners=[
        craft_runner,
    ],
)

@svc.api(input=Image(), output=JSON())
async def predict(f: PILImage) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
    assert isinstance(f, PILImage)
    # preprocessing
    image = np.array(f.convert('RGB'))
    height = image.shape[0] // 12
    images=[]   # 줄 단위 이미지
    for i in range(0, 10, 3):
        usr = image[height * (i + 2):height * (i + 3), :].copy()
        images.append(usr)
    # load data
    boxes = {}
    for k, img in enumerate(images):
        image = utils.imgproc(img)      # resize image and nomalization
        x = torch.from_numpy(image).permute(2, 0, 1)    # [h, w, c] to [c, h, w]

        y = await craft_runner.async_run(x)
        score_text = y[0,:,:,0].cpu().data.numpy()
        bbox = utils.getDetBoxes(score_text)
        boxes['line'+str(k)] = bbox

    return boxes