import xml.etree.ElementTree as ET
from collections import OrderedDict
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os

class HRSCDataset(Dataset):
    """
    HRSC Datasets, authored by Li Jinyu
    
    """
    CLASS = ('ship', )
    def __init__(self,
                 root_dir='./datasets/HRSC2016/',
                 mode='trainval'
                 ):
        super(HRSCDataset, self).__init__()
        self.root_dir = root_dir
        self.ann_subdir = os.path.join(root_dir, 'FullDataset', 'Annotations')
        self.img_subdir = os.path.join(root_dir, 'FullDataset', 'AllImages')
        self.mode = mode
        
    def load_annotations(self, ann_file):
        data_info = {}
        ann_file_path = os.path.join(self.ann_subdir, ann_file)
        tree = ET.parse(ann_file_path)
        root = tree.getroot()
        
        width = int(root.find('Img_SizeWidth').text)
        height = int(root.find('Img_SizeHeight').text)
        
        data_info['width'] = width
        data_info['height'] = height
        
        data_info['anns'] = {}
        
        gt_bboxes = []
        gt_labels = []
        
        for obj in root.findall('HRSC_Objects/HRSC_Object'):
            label = 0
            bbox = np.array([[
                    float(obj.find('mbox_cx').text) / width,
                    float(obj.find('mbox_cy').text) / height,
                    float(obj.find('mbox_w').text) / width,
                    float(obj.find('mbox_h').text) / height,
                    float(obj.find('mbox_ang').text)
                    ]], dtype=np.float32)
            gt_bboxes.append(bbox)
            gt_labels.append(label)
        if gt_bboxes:
            data_info['ann']['gt_bboxes'] = np.array(gt_bboxes)
            data_info['ann']['gt_labels'] = np.array(gt_label)
        
        return data_info
    
    