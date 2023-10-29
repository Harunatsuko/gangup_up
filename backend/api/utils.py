import numpy as np


def preprocess(rbg_img, ms_img, channel_num=11):
    arrs = []
    img = np.array(rbg_img)
    for channel in range(channel_num):
        ms_img.seek(channel)
        np_arr = np.array(ms_img)
        arrs.append(np_arr)
    out_arr = np.stack([img[:, :, 0], img[:, :, 1], img[:, :, 2]] + arrs)

    out_channels = []
    for channel_id in range(channel_num + 3):
        new_arr = np.zeros((640, out_arr.shape[2]))
        new_arr[0:img.shape[0], :] = out_arr[channel_id, :, :]
        out_channels.append(new_arr)
    out_arr = np.stack(out_channels)
    return out_arr


def from_yolo_to_cv(bbox, shape=None):
    # shape - (width, height, channels)
    x_top = max(0, bbox[1] - bbox[3] / 2)
    y_top = max(0, bbox[2] - bbox[4] / 2)
    x_bot = min(1, bbox[1] + bbox[3] / 2)
    y_bot = min(1, bbox[2] + bbox[4] / 2)

    if shape is not None:
        x_top = int(x_top * shape[1])
        y_top = int(y_top * shape[0])
        x_bot = int(x_bot * shape[1])
        y_bot = int(y_bot * shape[0])

    return [x_top, y_top], [x_bot, y_bot]