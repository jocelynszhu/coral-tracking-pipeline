from utils import *

from utils_yolo import *
from sort import Sort

import cv2

MODEL_NAME = 'models/1964_3.tflite'
MODEL_YAML = 'models/1964_3.yaml'

alg_info = {
    "classes": ['other_animal', 'pig', 'blackbear', 'bobcat', 'rabbit', 'cougar', 'skunk', 'otter', 'rat']
}
yolo = EdgeTPUModel(MODEL_NAME, alg_info)
input_size = yolo.get_image_size()
mot_tracker = Sort(max_age=1, 
                       min_hits=3,
                       iou_threshold=0.3)


def tracking(vid_path, dimension, input_size):
    for img, img_pil in load_one_SK_PIL(vid_path, dimension):
        print("loaded image")
        _, net_image, _ = get_image_tensor(img_pil, input_size[0])
        dets = yolo.predict(net_image) #list of obj detections
        callback(img, dets, mot_tracker, writer)
        print("tracked image")
    writer.release()


vid_path = "kittens.AVI"
out_path = "kittens_sort.mp4"
dim = 224

vidcap     = cv2.VideoCapture(vid_path)
fps  = int(vidcap.get(cv2.CAP_PROP_FPS))

writer = cv2.VideoWriter(out_path, 
        cv2.VideoWriter_fourcc(*'mp4v'), fps, (dim, dim))

tracking(vid_path, dim, input_size)