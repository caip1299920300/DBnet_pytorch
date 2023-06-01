# -*- coding: utf-8 -*-
# @Time    : 2019/8/24 12:06
# @Author  : zhoujun

import os
import sys
import pathlib
import numpy as np

__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

# project = 'DBNet.pytorch'  # 工作项目根目录
# sys.path.append(os.getcwd().split(project)[0] + project)
import time
import cv2
import torch

from data_loader import get_transforms
from models import build_model
from post_processing import get_post_processing


def resize_image(img, short_size):
    height, width, _ = img.shape
    if height < width:
        new_height = short_size
        new_width = new_height / height * width
    else:
        new_width = short_size
        new_height = new_width / width * height
    new_height = int(round(new_height / 32) * 32)
    new_width = int(round(new_width / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img


class Pytorch_model:
    def __init__(self, model_path, post_p_thre=0.7, gpu_id=None):
        '''
        初始化pytorch模型
        :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
        :param gpu_id: 在哪一块gpu上运行
        '''
        self.gpu_id = gpu_id

        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
        else:
            self.device = torch.device("cpu")
        print('device:', self.device)
        checkpoint = torch.load(model_path, map_location=self.device)

        config = checkpoint['config']
        config['post_processing']['args']['unclip_ratio'] = 1.4
        config['arch']['backbone']['pretrained'] = False
        self.model = build_model(config['arch'])
        self.post_process = get_post_processing(config['post_processing'])
        self.post_process.box_thresh = post_p_thre
        self.img_mode = config['dataset']['train']['dataset']['args']['img_mode']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.transform = []
        for t in config['dataset']['train']['dataset']['args']['transforms']:
            if t['type'] in ['ToTensor', 'Normalize']:
                self.transform.append(t)
        self.transform = get_transforms(self.transform)

    def predict(self, img_path: str, is_output_polygon=False, short_size: int = 1024):
        '''
        对传入的图像进行预测，支持图像地址,opecv 读取图片，偏慢
        :param img_path: 图像地址
        :param is_numpy:
        :return:
        '''
        assert os.path.exists(img_path), 'file is not exists'
        img = cv2.imread(img_path, 1 if self.img_mode != 'GRAY' else 0)
        img = cv2.resize(img, (args.size, args.size))
        if self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = np.rot90(img)
        # cv2.imshow('img',img)
        # img = resize_image(img, short_size)
        h, w = img.shape[:2]

        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = self.transform(img)
        tensor = tensor.unsqueeze_(0)

        tensor = tensor.to(self.device)
        batch = {'shape': [(h, w)]}
        with torch.no_grad():
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize()
            start = time.time()
            preds = self.model(tensor)
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize()
            box_list, score_list = self.post_process(batch, preds, is_output_polygon=is_output_polygon)
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
            t = time.time() - start
            print("time: ",t)
            # cv2.waitKey(0)
        return preds[0, 0, :, :].detach().cpu().numpy(), box_list, score_list, t


def save_depoly(model, input, save_path):
    traced_script_model = torch.jit.trace(model, input)
    traced_script_model.save(save_path)


def init_args():
    import argparse
    parser = argparse.ArgumentParser(description='DBNet.pytorch')
    parser.add_argument('--model_path', default=r'./output/DBNet_resnet18_FPN_DBHead/checkpoint/model_best.pth', type=str)
    parser.add_argument('--input_folder', default='./test/input', type=str, help='img path for predict')
    parser.add_argument('--output_folder', default='./test/output', type=str, help='img path for output')
    parser.add_argument('--thre', default=0.4,type=float, help='the thresh of post_processing')
    parser.add_argument('--polygon', default=False,type=bool, help='output polygon or box')
    parser.add_argument('--size', default=512, type=int, help='size')
    parser.add_argument('--show', action='store_true', help='show result')
    parser.add_argument('--save_result', action='store_true', help='save box and score to txt file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import pathlib
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from utils.util import show_img, draw_bbox, save_result, get_file_list

    args = init_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str('0')
    # 初始化网络
    model = Pytorch_model(args.model_path, post_p_thre=args.thre, gpu_id=0)
    img_folder = pathlib.Path(args.input_folder)
    for img_path in tqdm(get_file_list(args.input_folder, p_postfix=['.jpg','.JPG','.jpeg'])):
        preds, boxes_list, score_list, t = model.predict(img_path, is_output_polygon=args.polygon,short_size=args.size)
        # img = np.rot90(cv2.imread(img_path))
        img = cv2.imread(img_path)
        img = cv2.resize(img, (args.size, args.size))
        img = draw_bbox(img[:, :, ::-1], boxes_list)
        if args.show:
            show_img(preds)
            show_img(img, title=os.path.basename(img_path))
            plt.show()
        # 保存结果到路径
        os.makedirs(args.output_folder, exist_ok=True)
        img_path = pathlib.Path(img_path)
        output_path = os.path.join(args.output_folder, img_path.stem + '_result.jpg')
        pred_path = os.path.join(args.output_folder, img_path.stem + '_pred.jpg')

        # 二值化
        imgray = preds * 255
        ret, thresh = cv2.threshold(imgray, 125, 255, 0)

        contours, hierarchy = cv2.findContours(np.uint8(thresh), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 轮廓线提取矩形、最小外接矩形、四个点
        for c in contours:
            if len(c) < 10:  # 去除外轮廓
                continue
            # 找面积最小的矩形
            rect = cv2.minAreaRect(c)  # ((cx, cy), (bw, bh), angle), 浮点数
            # 得到最小矩形的坐标
            box = cv2.boxPoints(rect)  # 最小外接矩形的四个点坐标,浮点数
            # # 找到边界坐标
            # x, y, w, h = cv2.boundingRect(c)  # 计算点集最外面的矩形边界
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红

            # 找面积最小的矩形
            rect = cv2.minAreaRect(c)  # ((cx, cy), (bw, bh), angle), 浮点数
            # 得到最小矩形的坐标
            box = cv2.boxPoints(rect)  # 最小外接矩形的四个点坐标,浮点数
            # 标准化坐标到整数
            box = np.int0(box)  # 四个点坐标转为整数
            # 画出边界
            (bw, bh) = rect[1]  # 最小外接矩形宽高
            angle = rect[2]     # 最小外接矩形角度
            bsz = max(bw, bh) / min(bw, bh)  # 最长边比最短边，判断是否有细长物
            if bsz>10 and max(bw, bh)>min(img.shape[:2])*0.2: # angle 去掉竹子的干扰;并设置最小外接矩形最长边大于20像素
               cv2.drawContours(img, [box], 0, (0, 255, 0), 3)  # 以轮廓线形式画出最小外接矩形，绿色
            # cv2.drawContours(img, [box], 0, (0, 255, 0), 3)  # 以轮廓线形式画出最小外接矩形，绿色

            # # 最小外接圆
            # center, radius = cv2.minEnclosingCircle(c)
            # '''
            #     cv2.circle(img, center, radius, color, thickness, lineType,shift)
            # # 画圆
            # # img,被画圆的图像    center,圆心坐标 radius,圆半径
            # # color,圆的颜色,BGR颜色  thickness,线的宽度，-1的时候表示填充圆
            # # lineType,有三种cv2.LINE_AA、cv2.LINE_4、cv2.LINE_8
            # # cv2.LINE_AA抗锯齿，这种类型的线看起来平滑点
            # # shift,坐标精确到小数点后第几位
            # '''
            # cv2.circle(img, (int(center[0]), int(center[1])), int(radius), (255, 0, 0), 2)

        # img = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)  # img为三通道才能显示轮廓
        # cv2.imshow('drawimg', img)
        # cv2.waitKey(0)

        cv2.imwrite(pred_path, preds * 255)
        # cv2.imwrite(pred_path, img)
        # save_result(output_path.replace('_result.jpg', '.txt'), boxes_list, score_list, args.polygon)