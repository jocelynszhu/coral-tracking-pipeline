
import cv2
import numpy as np

import skvideo
import skvideo.io
import svgwrite

import colorsys

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
        video = loadVideoSK(path, n_frames, greyscale=False)
        video = video[::1,...]
        print("loaded_video")
        for frame in video:
            img = mold_image(frame[:,:,:], dim)
            im_pil = Image.fromarray(img)
            yield img, im_pil

def input_image_size(interpreter):
    """Returns input size as (width, height, channels) tuple."""
    _, height, width, channels = interpreter.get_input_details()[0]['shape']
    return width, height, channels


def callback(image, dim, objs, mot_tracker, writer):
    detections = []
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    print(objs)
    for obj in objs:
        x0, y0, x1, y1 = obj.bbox
        element = convert2bbox(x0, y0, x1, y1, dim)
        element.append(obj.score)  # print('element= ',element)
        detections.append(element)  # print('dets: ',dets)
    detections = np.array(detections)
    trdata = []
    trdata = mot_tracker.update(detections)

    for i in range(len(trdata.tolist())):
        coords = trdata.tolist()[i]
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        name_idx = int(coords[4])
        name = "ID: {}".format(str(name_idx))
        color = create_unique_color_float(name_idx)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        print("track ", name, "(", x1, y1,"), ", "(", x2, y2,")")
        cv2.putText(image, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness=2)
    writer.write(image)

def convert2bbox(x0, y0, x1, y1, dim):
    x = x0 + (x1 - x0) / 2
    y = y0 + (y1 - y0) / 2
    w = x1 - x0
    h = y1 - y0
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return [x1*dim, y1*dim, x2*dim, y2*dim]

# def yolobbox2bbox(x,y,w,h, dim):
#     x1, y1 = x-w/2, y-h/2
#     x2, y2 = x+w/2, y+h/2
#     return [x1, y1, x2, y2]

     
def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return int(255*r), int(255*g), int(255*b)