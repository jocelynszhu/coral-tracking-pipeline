
import cv2
import numpy as np

import skvideo
import skvideo.io

from PIL import Image

def mold_image(image, dimension):
    image_dtype = image.dtype
    resized = cv2.resize(image, (dimension, dimension), interpolation= cv2.INTER_AREA)
    print("done resizing")
    h, w = resized.shape[:2]
    top_pad = (dimension - h) // 2
    bottom_pad = dimension - h - top_pad
    left_pad = (dimension - w) // 2
    right_pad = dimension - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    padded = np.pad(resized, padding, mode="constant", constant_values=0)
    return padded.astype(image_dtype)

def load_one(vid_data, box_data, dim):
    for i in range(len(vid_data)):
        bboxes = box_data[i]
        molded_img = mold_image(vid_data[i], dimension=dim)
        yield molded_img, bboxes

def loadVideoSK(path, num_frames=None, greyscale=True):
    # load the video
    if not num_frames is None:
        return skvideo.io.vread(path, as_grey=greyscale, num_frames=num_frames)
    else:
        return skvideo.io.vread(path, as_grey=greyscale)
    
def load_one_CV_PIL(path, dim):
        vidcap = cv2.VideoCapture(path)
        while(True):
            ret,frame = vidcap.read()
            if ret:
                img = mold_image(frame, dim)
                im_pil = Image.fromarray(img)
                yield im_pil
            else:
                break

def load_one_SK_PIL(path, dim):
        vidcap = cv2.VideoCapture(path)
        n_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) # number of frames
        video = loadVideoSK(path, n_frames, grey_scale=False)
        video = video[::1,...]
        for frame in video:
            img = mold_image(frame[:,:,:], dim)
            im_pil = Image.fromarray(img)
            yield im_pil


