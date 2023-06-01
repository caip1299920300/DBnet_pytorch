# -*- coding:utf-8 -*-
'''
@Author: siyuan
@功能: 无
@输入参数: 无
@输出参数: 无
@注意要点: 无
'''

import torch
import torch.nn as nn
import numpy as np
import shutil
import matplotlib.pyplot as plt
import cv2
import os 

from models_pt.model import Model
from data_loader import get_transforms
# from post_processing import get_post_processing
from seg_detector_representer import SegDetectorRepresenter
from utils.util import show_img, draw_bbox, save_result, get_file_list
from datetime import datetime

# def predictByPt(img:np.ndarray, size=512):
#     img = cv2.resize(img, (size, size))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     h, w = img.shape[:2]
#
#     # 将图片由(w,h)变为(1,img_channel,h,w)
#     tensor = transform(img)
#     tensor = tensor.unsqueeze_(0)
#     tensor = tensor.to(device)
#     batch = {'shape': [(h, w)]}
#     with torch.no_grad():
#         preds = tsm(tensor)
#         preds = preds.cpu()
#         box_list, score_list = post_process(batch, preds, is_output_polygon=is_output_polygon)
#         box_list, score_list = box_list[0], score_list[0]
#         if len(box_list) > 0:
#             if is_output_polygon:
#                 idx = [x.sum() > 0 for x in box_list]
#                 box_list = [box_list[i] for i, v in enumerate(idx) if v]
#                 score_list = [score_list[i] for i, v in enumerate(idx) if v]
#             else:
#                 idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0  # 去掉全为0的框
#                 box_list, score_list = box_list[idx], score_list[idx]
#         else:
#             box_list, score_list = [], []


def pth2pt(pthPath:str, ptPath:str, testPath:str, size=512, is_output_polygon = False, unclip_ratio = 1.4,post_p_thre=0.7 ):
    device = 'cuda:0'
    checkpoint = torch.load(pthPath, map_location=device)

    config = checkpoint['config']
    config['post_processing']['args']['unclip_ratio'] = unclip_ratio
    config['arch']['backbone']['pretrained'] = False
    model = Model(config['arch'])
    post_process = SegDetectorRepresenter(thresh=0.3, box_thresh=0.7, max_candidates=1000, unclip_ratio=1.0)
    post_process.box_thresh = post_p_thre
    img_mode = config['dataset']['train']['dataset']['args']['img_mode']
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    transform = []
    for t in config['dataset']['train']['dataset']['args']['transforms']:
        if t['type'] in ['ToTensor', 'Normalize']:
            transform.append(t)
    transform = get_transforms(transform)

    # trace
    with torch.no_grad():
        exam = torch.rand(1, 3, 512, 512).cuda()
        # tsm = torch.jit.trace(net, (exam, adj))
        tsm = torch.jit.trace(model, example_inputs=exam)
        tsm.save(ptPath)
        print('== save = = ',ptPath)

    #== 测试 ===
    assert os.path.exists(testPath), 'file is not exists'
    img = cv2.imread(testPath, 1 if img_mode != 'GRAY' else 0)
    img = cv2.resize(img, (size, size))
    if img_mode == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # 将图片由(w,h)变为(1,img_channel,h,w)
    tensor = transform(img)
    tensor = tensor.unsqueeze_(0)
    tensor = tensor.to(device)
    batch = {'shape': [(h, w)]}
    with torch.no_grad():
        preds = tsm(tensor)
        preds = preds.cpu()
        box_list, score_list = post_process(batch, preds, is_output_polygon=is_output_polygon)
        box_list, score_list = box_list[0], score_list[0]
        if len(box_list) > 0:
            if is_output_polygon:
                idx = [x.sum() > 0 for x in box_list]
                box_list = [box_list[i] for i, v in enumerate(idx) if v]
                score_list = [score_list[i] for i, v in enumerate(idx) if v]
            else:
                idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0  # 去掉全为0的框
                box_list, score_list = box_list[idx], score_list[idx]
        else:
            box_list, score_list = [], []

        img = cv2.imread(testPath)
        img = cv2.resize(img, (size, size))
        img = draw_bbox(img[:, :, ::-1], box_list)
        cv2.imshow("pre",preds[0].detach().numpy())
        # cv2.imshow("pre_thre", preds[0, 1, :, :].detach().numpy())
        # cv2.imshow("pre_add", preds[0, 2, :, :].detach().numpy())
        cv2.imshow("img", img)
        cv2.waitKey(0)
        # show_img(preds)
        # show_img(img, title=os.path.basename(testPath))
        # plt.show()

    return preds.detach().cpu().numpy(), box_list, score_list
    

if __name__ == "__main__":
    pthPath = "./output/DBNet_resnet18_FPN_DBHead/checkpoint/model_best.pth"
    ptPath = "./output/DBNet_resnet18_FPN_DBHead/checkpoint/boardFindNet.pt"
    testPath = "./test/input/1484204144.00.jpg"
    pth2pt(pthPath, ptPath, testPath)