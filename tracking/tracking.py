import cv2
import numpy as np
from dataclasses import dataclass
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import LinearRegression
import onnxruntime as ort
from plot_utils import cells_to_bboxes, plot_image, non_max_suppression

@dataclass
class RecyclableObject:
    bbox: list
    idx: int
    next_pos: list
    ch_mean: np.array

def custom_distance(bbox, ch_mean, obj):
    ## сравниваем по размеру bbox
    ## сравниваем по среднему каналов
    ## сравниваем по расстоянию между центрами масс
    ## сравниваем по предсказанному центру масс и реальному
    bbox_area = bbox[3]*bbox[4]
    obj_bbox_area = obj.bbox[3]*obj.bbox[4]
    #0
    area_diff = 1 - min(bbox_area, obj_bbox_area) / max(bbox_area, obj_bbox_area)
    #1
    ch_diff = np.abs(ch_mean-obj.ch_mean).mean()
    #2
    distance = np.linalg.norm(np.array([bbox[1], bbox[2]])-np.array([[obj.bbox[1], obj.bbox[2]]]))
    #3
    if obj.next_pos is not None:
        pred_distance = np.linalg.norm(np.array([bbox[1], bbox[2]])-obj.next_pos)
    else:
        pred_distance = None
    
    return area_diff, ch_diff, distance, pred_distance


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
    
class Traker():
    def __init__(self,
                 detector_path,
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
        self.detector_path = detector_path
        self.is_staying = False
        self.stay_frames = 0
        self.linear_predictor = LinearPredictor()
        self.visualiser = Visualiser()
        self.session = ort.InferenceSession(self.detector_path)
    
    def _get_channel_means(self, bboxes, arr):
        ch_means = []
        for bbox in bboxes:
            top, bot = from_yolo_to_cv(bbox, (arr.shape[2],arr.shape[1]))
            
            bbox_arr = arr[:, top[0]:bot[0], top[1]:bot[1]]
            means = np.array([np.mean(bbox_arr[i, :, :]) for i in range(len(bbox_arr))])
            ch_means.append(means)
        return ch_means
            
    
    def track(self, frame_rgb, frame_ms, frame_idx, path_rgb):
        arr = preprocess(frame_rgb, frame_ms)
        bboxes = self._detect(arr)
        
        bboxes_channel_means = self._get_channel_means(bboxes,
                                                       arr)
        objs, is_stay_detected = self._update_tracks(bboxes, bboxes_channel_means, arr)
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
        
    def _update_tracks(self, bboxes, bboxes_channel_means, arr):
        stay_boxes = 0
        x_left_border = 0.98
        x_right_border = 0.02
        objs = []
        busy_obj_idxs = []
        for bbox, ch_mean in zip(bboxes, bboxes_channel_means):
            top, bottom = from_yolo_to_cv(bbox)
            if bottom[0] < x_left_border and top[0] > x_right_border:
                good_cand = []
                metrices = []
                for obj in self._current_objs:
                    area_diff, ch_diff, distance, pred_distance = custom_distance(bbox,ch_mean, obj)
                    if pred_distance is not None:
                        if pred_distance < 0.1:
                            if area_diff < 0.15 and ch_diff < 10 and distance < 0.2:
                                good_cand.append(obj)
                                metrices.append((distance+pred_distance)/2)
                                
                    else:
                        if area_diff < 0.1 and ch_diff < 10 and distance < 0.08:
                            good_cand.append(obj)
                            metrices.append(distance)
                                
                if len(good_cand) > 1:
                    max_m = 100
                    best_cand = None
                    for cand, m in zip(good_cand, metrices):
                        if m < max_m:
                            max_m = m
                            best_cand = cand
                            objs.append(RecyclableObject(bbox,
                                         best_cand.idx,
                                         self.linear_predictor.predict_next_pos(bbox,best_cand.idx),
                                         ch_mean))
                            if max_m < self.min_dist_threshold:
                                stay_boxes += 1
                elif len(good_cand) == 1:
                    best_cand = good_cand[0]
                    
                    objs.append(RecyclableObject(bbox,
                                         best_cand.idx,
                                         self.linear_predictor.predict_next_pos(bbox,best_cand.idx),
                                         ch_mean))
                    if metrices[0] < self.min_dist_threshold:
                        stay_boxes += 1
                else:
                    is_real_object = self.check_bbox_ch_means(bbox, arr, top, bottom)
                    if is_real_object:
                        objs.append(RecyclableObject(bbox,
                                         self._curr_id,
                                         self.linear_predictor.predict_next_pos(bbox,self._curr_id),
                                         ch_mean))
                        self._curr_id += 1
        return objs, stay_boxes == len(bboxes)
    
    
    def check_bbox_ch_means(self, bbox, arr, top, bot):
        w, h = arr.shape[0], arr.shape[1]
        bbox_left = [bbox[0],max(0,bbox[1]-0.1),bbox[2], bbox[3], bbox[4]]
        bbox_right = [bbox[0],min(w,bbox[1]+0.1),bbox[2], bbox[3], bbox[4]]
        ch_means = self._get_channel_means([bbox_left,bbox_right,bbox], arr)
        c1 = (np.mean(ch_means[0]-ch_means[1]) - np.mean(ch_means[0]-ch_means[2])) > 5
        c2 = (np.mean(ch_means[0]-ch_means[1]) - np.mean(ch_means[1]-ch_means[2])) > 5
        return c1 and c2
                
    def _detect(self, arr, thr=0.9):
        input_arr = arr/255
        outputs = self.session.run(None, {"input.1": np.float32([input_arr])})
        bboxes = cells_to_bboxes(outputs, is_pred=True, to_list=False)
        bboxes = non_max_suppression(bboxes, iou_threshold=0.01, threshold=thr, tolist=False)
        bboxes = bboxes.numpy()
        boxes_out = [[bbox[0]]+list(bbox[2:]/640) for bbox in bboxes]
        return boxes_out
    
    
# Usage
# traker = Traker('yolov5_v6_640_640.onnx')
# for i, framep in enumerate(frames_pths):
#     frame_rgb = Image.open(framep)
#     frame_ms = Image.open(framep.replace('frames_rgb',
#                                 'frames_ms').replace('png','tif'))
#     objs, rgb_img, gray_imgs = traker.track(frame_rgb,frame_ms,i, framep )