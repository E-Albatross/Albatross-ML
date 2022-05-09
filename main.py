import argparse

import bentoml
import torch

from process import CraftMain

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, help='enable CUDA')
    parser.add_argument("--detection_model", default="./model/craft_mlt_25k.pth")
    args = parser.parse_args()

    cuda = args.cuda and torch.cuda.is_available()

    craft = CraftMain()
    # load model
    model = craft.load_model(args.detection_model, cuda)

    # save model to BentoML's standard format in a local model store
    tag = bentoml.pytorch.save(
        "craft",
        model
    )


