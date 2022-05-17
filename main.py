import argparse

import bentoml

from process import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, help='enable CUDA')
    parser.add_argument("--craft_model", default="./model/craft_mlt_25k.pth", help='text detection model path')
    parser.add_argument("--fpn_model", default="./model/fpn_last.pth", help='segmentation model path')
    parser.add_argument("--seg_model", default="./model/seg_model.pth", help='segmentation model path')


    args = parser.parse_args()

    cuda = args.cuda and torch.cuda.is_available()

    craft = CraftMain()
    fpn = FPNMain()
    seg = SegmentationMain()

    # load model
    craft_model = craft.load_model(args.craft_model, cuda)
    fpn_model = fpn.load_model(args.fpn_model, cuda)
    segmentation_model = seg.load_model(args.seg_model, cuda)

    # save model to BentoML's standard format in a local model store
    bentoml.pytorch.save(
        "craft",
        craft_model
    )
    bentoml.pytorch.save(
        "fpn",
        fpn_model
    )
    bentoml.pytorch.save(
        "segmentation",
        segmentation_model
    )


