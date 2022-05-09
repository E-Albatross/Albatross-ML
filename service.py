import logging

import bentoml

import numpy as np
import torch

from bentoml.io import Image, NumpyNdarray
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

@svc.api(input=Image(), output=NumpyNdarray(dtype="int64"))
async def predict(f: PILImage) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
    assert isinstance(f, PILImage)
    # preprocessing
    image = f.convert('RGB')
    arr = np.array(image)
    image = utils.imgproc(arr)      # resize image and nomalization

    x = torch.from_numpy(image).permute(2, 0, 1)    # [h, w, c] to [c, h, w]

    logger.info(f'x shape = {x.shape}')     # print input shape -- for test

    y = await craft_runner.async_run(x)

    score_text = y[0,:,:,0].cpu().data.numpy()

    boxes = utils.getDetBoxes(score_text)

    return boxes