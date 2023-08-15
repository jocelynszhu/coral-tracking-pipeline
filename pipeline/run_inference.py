from utils import *
import cv2
import pycoral.utils.edgetpu as etpu

def processing(video, dimension, interpreter, input_details, output_details):
    i = 0
    tracking_frames = []
    for img, img_pil in load_one_SK_PIL(video, dimension):
        if len(tracking_frames) < 11:
            tracking_frames.append(img)
        else:
            inference_frames = np.asarray(tracking_frames, dtype=np.float32)
            prediction = behavior(inference_frames, interpreter, input_details, output_details)
            print(prediction)
            tracking_frames = []
        print("loaded image ", i)
        i+=1


if __name__ =="__main__":
    vid_path = "limping.mp4"
    dim = 224

    model_name = 'models/1964_3.tflite'
    interpreter_name = 'models/behavioral_model.tflite'
    
    interpreter = etpu.make_interpreter(interpreter_name)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    vidcap     = cv2.VideoCapture(vid_path)
    fps  = int(vidcap.get(cv2.CAP_PROP_FPS))
    n_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) # number of frames
    video = loadVideoSK(vid_path, n_frames, greyscale=False)
    video = video[::1,...]

    print("loaded behavioral interpreter")
    processing(video, dim, interpreter, input_details, output_details)



