from utils import load_one_CV_PIL

from utils_yolo import *

MODEL_NAME = 'models/1964_3.tflite'
MODEL_YAML = 'models/1964_3.yaml'

alg_info = {
    "classes": ['other_animal', 'pig', 'blackbear', 'bobcat', 'rabbit', 'cougar', 'skunk', 'otter', 'rat']
}
edgetpu = EdgeTPUModel(MODEL_NAME, alg_info)
input_size = edgetpu.get_image_size()


def tracking(vid_path, dimension, input_size):
    for img in load_one_CV_PIL(vid_path, dimension):
        full_image, net_image, pad = get_image_tensor(img, input_size[0])
        pred = edgetpu.forward(net_image)
        bbox = edgetpu.process_predictions(pred[0], full_image, pad)
        print(bbox)

vid_path = "kittens.AVI"
dim = 224

tracking(vid_path, dim, input_size)