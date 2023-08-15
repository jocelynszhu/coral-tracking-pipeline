from utils import *

from utils_yolo import *
from sort import Sort

import cv2
import pycoral.utils.edgetpu as etpu



def tracking(video, dimension, input_size, writer):
    print("began tracking")
    mot_tracker = Sort(max_age=1, 
                       min_hits=3,
                       iou_threshold=0.3)

    i = 0
    tracklets = {}
    for img, img_pil in load_one_SK_PIL(video, dimension):
        print("loaded image ", i)
        try:
            _, net_image, _ = get_image_tensor(img_pil, input_size[0])
            dets = yolo.predict(net_image) #list of obj detections
            track(img, i, dimension, dets, mot_tracker, writer, tracklets)
            print("tracked image")
        except:
            pass
        i += 1
    writer.release()
    return tracklets


def init_pipeline(vid_path, out_path, detector_file, dim):

    vidcap     = cv2.VideoCapture(vid_path)
    fps  = int(vidcap.get(cv2.CAP_PROP_FPS))
    n_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) # number of frames
    video = loadVideoSK(vid_path, n_frames, greyscale=False)
    video = video[::1,...]
    
    print("loaded video")

    writer = cv2.VideoWriter(out_path, 
            cv2.VideoWriter_fourcc(*'mp4v'), fps, (dim, dim))

        
    alg_info = {
        "classes": ['other_animal', 'pig', 'blackbear', 'bobcat', 'rabbit', 'cougar', 'skunk', 'otter', 'rat']
    }
    yolo = EdgeTPUModel(detector_file, alg_info, conf_thresh=0.4, iou_thresh=0.3)
    print("loaded yolo")


    print("initialized models and video")
    return video, writer, yolo


if __name__ =="__main__":
    vid_path = "kittens.AVI"
    out_path = "kittens_sort.mp4"
    dim = 224

    model_name = 'models/1964_3.tflite'
    interpreter_name = 'models/behavioral_model.tflite'

    video, writer, yolo= init_pipeline(vid_path, out_path, model_name, dim)

    tracklets_all = tracking(video, dim, yolo.get_image_size(), writer)

    yolo = None #free space 
    
    interpreter = etpu.make_interpreter(interpreter_name)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    

    print("loaded behavioral interpreter")

    for id, tracklets in tracklets_all.items():
        if len(tracklets_all[id]) >= 11:
            tracklets_id = tracklets_all[id][0:11] #figure out how to deal with longer
        else:
            continue
        tracklet_video = generate_tracklets(video, tracklets_id, dim)
        print("generated tracklet for ", id)
        id_behavior = behavior(tracklet_video, interpreter, input_details, output_details)
        print(id_behavior)

