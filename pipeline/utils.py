
import cv2
import numpy as np

import skvideo
import skvideo.io


import colorsys

from skimage.transform import rescale

from PIL import Image

def mold_image(image, dimension):
    """
    Resize image to dimension x dimension
    """
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
    """
    Generator yielding video frame by frame and bboxes corresponding to the frame

    parameters
        vid_data: np array formatted video
        box_data: np array formatted bounding boxes
        dim: dimension to rescale video frames to

    yields
        reshaped frame and corresponding bounding boxes
    """
    for i in range(len(vid_data)):
        bboxes = box_data[i]
        molded_img = mold_image(vid_data[i], dimension=dim)
        yield molded_img, bboxes

def loadVideoSK(path, num_frames=None, greyscale=True):
    """
    Intializes video from vid path to np array formatted video
    """
    # load the video
    if not num_frames is None:
        return skvideo.io.vread(path, as_grey=greyscale, num_frames=num_frames)
    else:
        return skvideo.io.vread(path, as_grey=greyscale)
    
def load_one_CV_PIL(path, dim):
    """
    Yields reshaped video frame by frame in PIL format with openCV (unused)
    """
    vidcap = cv2.VideoCapture(path)
    while(True):
        ret,frame = vidcap.read()
        if ret:
            img = mold_image(frame, dim)
            im_pil = Image.fromarray(img)
            yield im_pil
        else:
            break

def load_one_SK_PIL(video, dim):
    """
    Yields reshaped video frame by frame in both np array and PIL format
    """
    print("loaded_video")
    for frame in video:
        img = mold_image(frame[:,:,:], dim)
        im_pil = Image.fromarray(img)
        yield img, im_pil

def input_image_size(interpreter):
    """
    Returns behavioral model's input size as (width, height, channels) tuple
    """
    _, height, width, channels = interpreter.get_input_details()[0]['shape']
    return width, height, channels


def track(img, frame_idx, dim, objs, mot_tracker, writer, tracklets):
    """
    Tracks video frame, writes to final video and updates tracklet dictionary

    parameters
        img: reshaped video frame
        frame_idx: index of img in video
        dim: dimensions of video writer and reshaped img
        obj: object detections from yolo model
        mot_tracker: object tracker
        writer: video writer
        tracklets: dictionary mapping object id to relevant bounding boxes
            tracklets[object_id] = [[x1, y1, x2, y2, frame_idx]...]
    
    returns
        none
    """
    detections = []
    
    #format yolo detects for tracking
    for obj in objs:
        x0, y0, x1, y1 = obj.bbox
        element = convert2bbox(x0, y0, x1, y1, dim)
        element.append(obj.score)  # print('element= ',element)
        detections.append(element)  # print('dets: ',dets)
    detections = np.array(detections)
    trdata = []
    trdata = mot_tracker.update(detections)
    image =  cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #determine tracklets
    for i in range(len(trdata.tolist())):
        coords = trdata.tolist()[i]
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        name_idx = int(coords[4])
        coords[4] = frame_idx
        if name_idx in tracklets:
            tracklets[name_idx].append(coords)
        else:
            tracklets[name_idx] = [coords]
        image = displayBoxes(image, [x1, y1, x2, y2], name_idx)
    writer.write(image)

def convert2bbox(x0, y0, x1, y1, dim):
    """
    Converts coordinates from yolo bounding box to mask for image

    returns
        x1, y1, x2, y2 where (x1, y1) is the top left corner of bounding box and (x2, y2) is the bottom right
    """
    x = x0 + (x1 - x0) / 2
    y = y0 + (y1 - y0) / 2
    w = x1 - x0
    h = y1 - y0
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return [x1*dim, y1*dim, x2*dim, y2*dim]


def generate_tracklets(viddata, tracklets, dim): 
    """
    generates np array format of cropped tracklets
    
    parameters
        viddata: np array formatted video
        tracklets: tracklet dictionary mapping relevant bounding boxes to object_id
        dim: dimensions for tracklets for behavioral model

    returns
        (1, 33, dim, dim) shaped np array of cropped tracklets
    """
    tracklet_vid = []
    for tracklet_info in tracklets:
        mask = [int(tracklet_info[0]), int(tracklet_info[1]), int(tracklet_info[2]), int(tracklet_info[3])]
        frame = viddata[tracklet_info[4]]
        frame = mold_image(frame[:,:,:], dim)
        masked_frame = pad_img(mask, frame, dim)
        tracklet_vid.append(masked_frame)
    return np.asarray(tracklet_vid, dtype=np.float32).reshape(1, 33, dim, dim)

def create_unique_color_float(tag, hue_step=0.41):
    """
    Create a unique RGB color code for a given track id (tag).
    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return int(255*r), int(255*g), int(255*b)

def pad_img(mask, frame, dim):
    """
    returns frame cropped by mask [x1, y1, x2, y2] with size (dim, dim, 3)
    """
    rectsize = [mask[3] - mask[1], mask[2] - mask[0]]

    rectsize = np.asarray(rectsize)

    cutout = frame[mask[0] : mask[0] + rectsize[1], mask[1] : mask[1] + rectsize[0], :]

    padded_img = np.zeros((dim, dim, 3))

    padded_img[
        int(dim / 2 - cutout.shape[0] / 2) : int(
            dim / 2 + cutout.shape[0] / 2
        ),
        int(dim / 2 - cutout.shape[1] / 2) : int(
            dim / 2 + cutout.shape[1] / 2
        ),
        :,
    ] = cutout

    return padded_img

def displayBoxes(frame, mask, id, animal_id=None, mask_id=None):
    """
    updates frame with bounding box, object id, and behavioral id
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 1

    color = create_unique_color_float(id)
    cv2.rectangle(frame, (mask[1], mask[0]), (mask[3], mask[2]), color, 3)

    if animal_id:
        cv2.putText(
            frame,
            str(animal_id),
            (mask[1], mask[0]),
            font,
            0.7,
            color,
            font_thickness,
            cv2.LINE_AA,
        )

    return frame

def behavior(tracklets, interpreter, input_details, output_details):
    """
    runs inference on tracklets 
    returns behavioral predictions from interpreter
    """
    print(tracklets.shape)
    interpreter.set_tensor(input_details[0]['index'], tracklets.transpose(0,2,3,1))
    interpreter.set_tensor(input_details[1]['index'], tracklets.transpose(0,2,3,1)[...,:3])
    print("set tensors")
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("got prediction")
    return output_data