import numpy as np

def preprocess(rbg_img, ms_img, channel_num=11):
    arrs = []
    img = np.array(rbg_img)
    for channel in range(channel_num):
        ms_img.seek(channel)
        np_arr = np.array(ms_img)
        arrs.append(np_arr)
    out_arr = np.stack([img[:,:,0],img[:,:,1],img[:,:,2]]+arrs)
#     new_arr = np.zeros((channel_num+3,640, 640))
#     new_arr[:, 0:360, :] = out_arr
    return out_arr

def from_yolo_to_cv(bbox, shape = None):
    # shape - (width, height, channels)
    x_top = bbox[1]-bbox[3]/2
    y_top = bbox[2]-bbox[4]/2
    x_bot = bbox[1]+bbox[3]/2
    y_bot = bbox[2]+bbox[4]/2
    
    if shape is not None:
        x_top = int(x_top*shape[1])
        y_top = int(y_top*shape[0])
        x_bot = int(x_bot*shape[1])
        y_bot = int(y_bot*shape[0])
    
    return [x_top, y_top], [x_bot, y_bot]