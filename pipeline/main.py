from utils import load_one_CV_PIL, load_one_SK_PIL

from utils_yolo import *

MODEL_NAME = 'models/1964_3.tflite'
MODEL_YAML = 'models/1964_3.yaml'

alg_info = {
    "classes": ['other_animal', 'pig', 'blackbear', 'bobcat', 'rabbit', 'cougar', 'skunk', 'otter', 'rat']
}
edgetpu = EdgeTPUModel(MODEL_NAME, alg_info)
input_size = edgetpu.get_image_size()


def tracking(vid_path, dimension, input_size):
    for img in load_one_SK_PIL(vid_path, dimension):
        _, net_image, _ = get_image_tensor(img, input_size[0])
        bbox = edgetpu.predict(net_image)
        print(bbox)

vid_path = "kittens.AVI"
dim = 224

tracking(vid_path, dim, input_size)