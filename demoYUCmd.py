import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import argparse
from lib.core.config import  update_cfg
from lib.utils.utils import prepare_output_dir
from lib.models.e2e_model import Det_Track_VIBE

def runDemo(image_path, output_path, pretrained):
    model = Det_Track_VIBE()
    res = model.inferData(image_path, output_path, pretrained)
    model.renderRes(image_path, output_path, res)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    image_path = args.image_path
    pretrained = args.pretrained
    output_path = args.output_path
    runDemo(image_path, output_path, pretrained)