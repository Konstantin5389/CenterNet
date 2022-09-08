import torch.utils.data as data
import cv2
import numpy as np
import math
from .draw_gaussion import draw_umich_gaussian, gaussion_radius
from .transforms import random_flip, load_affine_matrix, random_crop_info, ex_box_jaccard
from . import data_augment
import torch

class BaseDataset(data.Dataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=None) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.phase = phase
        self.input_h = input_h
        self.input_w = input_w
        self.down_ratio = down_ratio
        self.img_ids = None
        self.num_classes = None
        self.max_objs = 500
        self.image_distort = data_augment.PhotometricDistort()
        
    def load_img_ids(self):
        """
        load img ids list
        """
        return None
    
    def load_img(self, idx):
        """
        from id load img
        """
        return None
    
    def load_annotation(self, idx):
        """
        Return
        {
            'pts': np.array [bl, tl, tr, br], dtype=float
            'cat': np.array [class_index], dtype=int
        }distort
        """
        return None
    
    def dec_evaluation(self, result_path):
        return None
    
    def data_transform(self, image, annotation):
        crop_size = None
        crop_center = None
        crop_size, crop_center = random_crop_info(h=image.shape[0], w=image.shape[1])
        image, annotation['pts'], crop_center = random_flip(image, annotation['pts'], crop_center)
        if crop_center == None:
            crop_center = np.asarray([image.shape[0] / 2, image.shape[1] / 2], dtype=np.float32)
        if crop_size == None:
            crop_size = [max(image.shape[0], image.shape[1]), max(image.shape[0], image.shape[1])]
        M = load_affine_matrix(crop_center=crop_center,
                               crop_size=crop_size,
                               dst_size=(self.input_w, self.input_h),
                               inverse=False,
                               rotation=True)
        image = cv2.warpAffine(src=image, M=M, dsize=(self.input_w, self.input_h), flags=cv2.INTER_LINEAR)
        if annotation['pts'].shape[0]:
            annotation['pts'] = np.concatenate([annotation['pts'], np.ones((annotation['pts'].shape[0], annotation['pts'].shape[1], 1))], axis=2)
            annotation['pts'] = np.matmul(annotation['pts'], np.transpose(M))
            annotation['pts'] = np.asarray(annotation['pts'], np.float32)

        out_annotations = {}
        size_thresh = 3
        out_rects = []
        out_cat = []
        for pt_old, cat in zip(annotation['pts'] , annotation['cat']):
            if (pt_old<0).any() or (pt_old[:,0]>self.input_w-1).any() or (pt_old[:,1]>self.input_h-1).any():
                pt_new = pt_old.copy()
                pt_new[:,0] = np.minimum(np.maximum(pt_new[:,0], 0.), self.input_w - 1)
                pt_new[:,1] = np.minimum(np.maximum(pt_new[:,1], 0.), self.input_h - 1)
                iou = ex_box_jaccard(pt_old.copy(), pt_new.copy())
                if iou>0.6:
                    rect = cv2.minAreaRect(pt_new/self.down_ratio)
                    if rect[1][0]>size_thresh and rect[1][1]>size_thresh:
                        out_rects.append([rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]])
                        out_cat.append(cat)
            else:
                rect = cv2.minAreaRect(pt_old/self.down_ratio)
                if rect[1][0]<size_thresh and rect[1][1]<size_thresh:
                    continue
                out_rects.append([rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]])
                out_cat.append(cat)
        out_annotations['rect'] = np.asarray(out_rects, np.float32)
        out_annotations['cat'] = np.asarray(out_cat, np.uint8)
        return image, out_annotations

    def __len__(self):
        return len(self.img_ids)

    def processing_test(self, image, input_h, input_w):
        image = cv2.resize(image, (input_w, input_h))
        out_image = image.astype(np.float32) / 255.
        out_image = out_image - 0.5
        out_image = out_image.transpose(2, 0, 1).reshape(1, 3, input_h, input_w)
        out_image = torch.from_numpy(out_image)
        return out_image
    
    def generate_ground_truth(self, image, annotation):
        image = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)
        image = self.image_distort(np.asarray(image, np.float32))
        image = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)
        image = np.transpose(image / 255. - 0.5, (2, 0, 1))
        
        image_h = self.input_h // self.down_ratio
        image_w = self.input_w // self.down_ratio
        
        hm = np.zeros((self.num_classes, image_h, image_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        ang = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        num_objs = min(annotation['rect'].shape[0], self.max_objs)
        
        for k in range(num_objs):
            cen_x, cen_y, bbox_w, bbox_h, theta = annotation['rect'][k, :]
            radius = gaussion_radius((math.ceil(bbox_h), math.ceil(bbox_w)))
            radius = max(0, int(radius))
            ct = np.asarray([cen_x, cen_y], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(hm[annotation['cat'][k]], ct_int, radius)
            ind[k] = ct_int[1] * image_w + ct_int[0]
            reg[k] = ct - ct_int
            reg_mask[k] = 1
            wh[k] = np.asarray([bbox_w, bbox_h], dtype=np.float32)
            ang[k] = np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float32)
        
        ret = {
            'input': image,
            'hm': hm,
            'reg_mask': reg_mask,
            'ind': ind,
            'wh': wh,
            'ang': ang,
            'reg': reg
        }
        return ret
    
    def __getitem__(self, index):
        image = self.load_img(index)
        image_h, image_w, _ = image.shape
        
        if self.phase == 'test':
            img_id = self.img_ids[index]
            image = self.processing_test(image, self.input_h, self.input_w)
            return {'image': image,
                    'img_id': img_id,
                    'image_w': image_w,
                    'image_h': image_h}
        
        elif self.phase == 'train':
            annotation = self.load_annotation(index)
            image, annotation = self.data_transform(image, annotation)
            data_dict = self.generate_ground_truth(image, annotation)
            return data_dict