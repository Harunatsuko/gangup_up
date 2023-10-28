import os
import numpy as np
import cv2
from dataclasses import dataclass
import matplotlib.pyplot as plt
from utils import preprocess

@dataclass
class RecyclableObject:
    bbox: list
    idx: int
    next_pos: list
    ch_mean: np.array

def custom_distance(bbox, ch_mean, obj, coeffs = [0.3, 0.2, 0.2, 0.3]):
    ## сравниваем по размеру bbox
    ## сравниваем по среднему каналов
    ## сравниваем по расстоянию между центрами масс
    ## сравниваем по предсказанному центру масс и реальному
    bbox_area = bbox[3]*bbox[4]
    obj_bbox_area = obj.bbox[3]*obj.bbox[4]
    area_diff = min(bbox_area, obj_bbox_area) / max(bbox_area, obj_bbox_area)
    
    ch_diff = np.abs(ch_mean-obj.ch_mean).mean()
    
    distance = np.linalg.norm(np.array([bbox[1], bbox[2]])-np.array([[obj.bbox[1], obj.bbox[2]]]))
    
    if obj.next_pos is not None:
        pred_distance = np.linalg.norm(np.array([bbox[1], bbox[2]])-obj.next_pos)
    else:
        pred_distance = 0
        correct_val = coeffs[-1]/3
        coeffs = [coeff+correct_val for coeff in coeffs]
        coeffs[-1] = 0
    
    return coeffs[0]*area_diff+coeffs[1]*ch_diff+coeffs[2]*distance+coeffs[3]*pred_distance


class LinearPredictor():
    def __init__(self, learn_stats_thresh=3):
        self.learn_stats_thresh = learn_stats_thresh
        self._learn = True
        self.stats = dict()
        
    def predict_next_pos(self, bbox, idx):
        if self._learn:
            if idx in self.stats.keys():
                self.stats[idx].append((bbox[1], bbox[2]))
            else:
                self.stats[idx] = [(bbox[1], bbox[2])]
            max_stats_len = max(self.stats, key=lambda l: len(self.stats[l]))
            if max_stats_len > self.learn_stats_thresh:
                self._learn = False
                self.fit_model()
            return None
        else:
            return self.predict(bbox[1], bbox[2])
        
    def fit_model(self):
        steps = []
        target = []
        for k, v in self.stats.items():
            if len(v)> 1:
                # save y for potential use
                xc, yc = v[0]
                for x, y in v[1:]:
                    steps.append([xc, yc])
                    target.append([x, y])
        model = LinearRegression()
        model.fit(steps, target)
        self.model = model
        
    def predict(self, x, y):
        pred = self.model.predict([[x,y]])
        return pred[0]
    
    def clear_stats(self):
        self.stats = dict()
        self._learn = False
        
    def start_learn(self):
        self._learn = True
        
        
class Traker():
    def __init__(self,
                 detector_path = '',
                 stay_frames_thresh=3,
                 min_dist_threshold = 0.01,
                 max_dist_threshold = 0.3,
                 is_draw_bbox_rgb = True,
                 is_draw_bbox_tif = False):
        self.max_dist_threshold = max_dist_threshold
        self.min_dist_threshold = min_dist_threshold
        self.is_draw_bbox_rgb = is_draw_bbox_rgb
        self.is_draw_bbox_tif = is_draw_bbox_tif
        self.stay_frames_thresh = stay_frames_thresh
        self._current_objs = []
        self._curr_id = 0
#         self.detector = self._load_detector(detector_path)
        self.is_staying = False
        self.stay_frames = 0
        self.linear_predictor = LinearPredictor()
        self.visualiser = Visualiser()
    
    def _get_channel_means(self, bboxes, arr):
        ch_means = []
        for bbox in bboxes:
            top, bot = from_yolo_to_cv(bbox, arr.shape[1:3])
            bbox_arr = arr[:,top[0]:bot[0], top[1]:bot[1]]
            means = np.array([np.mean(bbox_arr[i, :, :]) for i in range(len(bbox_arr))])
            ch_means.append(means)
        return ch_means
            
    
    def track(self, frame_rgb, frame_ms, frame_idx):
        # TODO: test stage
        bboxes = self._detect(frame_rgb, frame_ms, frame_idx)
        arr = preprocess(frame_rgb, frame_ms)
        bboxes_channel_means = self._get_channel_means(bboxes,
                                                       arr)
        objs, is_stay_detected = self._update_tracks(bboxes, bboxes_channel_means)
        self._stay_check(is_stay_detected)
        self._current_objs = objs
        if self.is_draw_bbox_rgb:
            rgb_img = self.visualiser.draw_bbox_rgb(frame_rgb, self._current_objs)
        else:
            rgb_img= None
        if self.is_draw_bbox_tif:
            tif_imgs = self.visualiser.draw_tif(frame_ms, self._current_objs)
        else:
            tif_imgs = self.visualiser.draw_tif(frame_ms)
        out_boxes = [[obj.idx, obj.bbox] for obj in self._current_objs]
        return out_boxes, rgb_img, tif_imgs
        
    def _stay_check(self, is_stay_detected):
        if is_stay_detected:
            if self.is_staying:
                self.stay_frames +=1
            else:
                if self.stay_frames >= self.stay_frames_thresh:
                    self.is_staying = True
                    self.linear_predictor.clear_stats()
        else:
            if self.is_staying:
                self.is_staying = False
                self.stay_frames = 0
                self.linear_predictor.start_learn()
        
    def _update_tracks(self, bboxes, bboxes_channel_means):
        stay_boxes = 0
        objs = []
        busy_obj_idxs = []
        for bbox, ch_mean in zip(bboxes, bboxes_channel_means):
#             print(bbox)
            distance = [(obj, custom_distance(bbox,ch_mean, obj)) for obj in self._current_objs \
                        if obj.idx not in busy_obj_idxs]
            if len(distance):
                distance = min(distance, key = lambda d: d[1])
                print('Distance', distance[1])
                if distance[1] > self.max_dist_threshold:
                    print('Found new object!')
                    objs.append(RecyclableObject(bbox,
                                     self._curr_id,
                                     self.linear_predictor.predict_next_pos(bbox,self._curr_id),
                                     ch_mean))
                    self._curr_id += 1
                else:
                    objs.append(RecyclableObject(bbox,
                                     distance[0].idx,
                                     self.linear_predictor.predict_next_pos(bbox,distance[0].idx),
                                     ch_mean))
                    busy_obj_idxs.append(distance[0].idx)
                    if distance[1] < self.min_dist_threshold:
                        stay_boxes += 1
            else:
                print('No object to compare')
                objs.append(RecyclableObject(bbox,
                                     self._curr_id,
                                     self.linear_predictor.predict_next_pos(bbox,self._curr_id),
                                     ch_mean))
                self._curr_id += 1
                
        return objs, stay_boxes == len(bboxes)
                
                
    def _detect(self, _frame_rgb, _frame_ms, frame_idx):
        bboxes = [[np.random.randint(4), 0.2, 0.1, 0.3, 0.1],
                  [np.random.randint(4), 0.3, 0.2, 0.4, 0.1]]
        return bboxes
    
class Visualiser():
    def __init__(self, class_colors=None):
        if class_colors is None:
            class_colors = {0:(255,0,0),
                            1:(0,255,0),
                            2:(0,0,255),
                            3:(150,0,150)}
        self.class_colors = class_colors
        
    def draw_bbox_rgb(self, frame, objects):
        frame = np.array(frame)
        for obj in objects:
            print(obj)
            class_label = obj.bbox[0]
            top, bot = from_yolo_to_cv(obj.bbox, frame.shape)
            frame = cv2.rectangle(frame, top, bot, self.class_colors[class_label], 2)
            frame = cv2.putText(frame, str(obj.idx), [top[0], bot[1]],
                            cv2.FONT_HERSHEY_SIMPLEX, 1, self.class_colors[class_label],2)
        return frame
    
    def draw_tif(self, frame, objects=None, channel_num=11):
        gray_imgs = []
        for channel in range(channel_num):
            frame.seek(channel)
            gray_img = np.uint8(np.array(frame)*255)
            if objects is not None:
                for obj in objects:
                    class_label = obj.bbox
                    top, bot = from_yolo_to_cv(obj.bbox)
                    gray_img = cv2.rectangle(gray_img, top, bot,self.class_colors[class_label], 2)
                    gray_img = cv2.putText(gray_img, str(obj.idx), [top[0], bot[1]],
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.class_colors[class_label],2)
            gray_imgs.append(gray_img)
        return gray_imgs
    
    
# for i, framep in enumerate(frames_pths):
#     frame_rgb = Image.open(framep)
#     frame_ms = Image.open(framep.replace('frames_rgb',
#                                 'frames_ms').replace('png','tif'))
#     objs, rgb_img, gray_imgs = traker.track(frame_rgb,frame_ms,i)
#     fig, ax = plt.subplots(figsize=(7,7))
    
#     ax.imshow(rgb_img)
#     plt.show()