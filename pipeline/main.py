from utils import load_one_CV_PIL

from utils_yolo import *

MODEL_NAME = 'model/1964_3.tflite'
MODEL_YAML = 'model/1964_3.yaml'

alg_info = {
    "classes": ['other_animal', 'pig', 'blackbear', 'bobcat', 'rabbit', 'cougar', 'skunk', 'otter', 'rat']
}
edgetpu = EdgeTPUModel(MODEL_NAME, alg_info)
input_size = edgetpu.get_image_size()


def tracking(vid_path, dimension, input_size):
    for img in load_one_CV_PIL(vid_path, dimension):
        _, net_image, _ = get_image_tensor(img, input_size[0])
        bbox = edgetpu.predict(net_image)
        print(bbox)
