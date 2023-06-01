# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:58
# @Author  : zhoujun
import time

import torch
import torchvision.utils as vutils
from tqdm import tqdm
import numpy as np
import cv2
from base import BaseTrainer
from utils import WarmupPolyLR, runningScore, cal_text_score
# 计算损失引用的函数
from data_loader.modules.make_shrink_map import MakeShrinkMap
from data_loader.modules.make_border_map import MakeBorderMap
from models.losses.basic_loss import BalanceCrossEntropyLoss, MaskL1Loss, DiceLoss

class Trainer(BaseTrainer):
    def __init__(self, config, model, criterion, train_loader, validate_loader, metric_cls, post_process=None):
        super(Trainer, self).__init__(config, model, criterion)
        self.show_images_iter = self.config['trainer']['show_images_iter']
        self.train_loader = train_loader
        if validate_loader is not None:
            assert post_process is not None and metric_cls is not None
        self.validate_loader = validate_loader
        self.post_process = post_process
        self.metric_cls = metric_cls
        self.train_loader_len = len(train_loader)
        self.MakeShrinkMap = MakeShrinkMap(min_text_size=3, shrink_ratio=0.8)  # 新增的特征图生成
        self.MakeBorderMap = MakeBorderMap(shrink_ratio=0.3, thresh_min=0.3, thresh_max=0.7)  # 新增的特征图生成
        self.eval_save_loss = 1e5
        if self.config['lr_scheduler']['type'] == 'WarmupPolyLR':
            warmup_iters = config['lr_scheduler']['args']['warmup_epoch'] * self.train_loader_len
            if self.start_epoch > 1:
                self.config['lr_scheduler']['args']['last_epoch'] = (self.start_epoch - 1) * self.train_loader_len
            self.scheduler = WarmupPolyLR(self.optimizer, max_iters=self.epochs * self.train_loader_len,
                                          warmup_iters=warmup_iters, **config['lr_scheduler']['args'])
        if self.validate_loader is not None:
            self.logger_info(
                'train dataset has {} samples,{} in dataloader, validate dataset has {} samples,{} in dataloader'.format(
                    len(self.train_loader.dataset), self.train_loader_len, len(self.validate_loader.dataset), len(self.validate_loader)))
        else:
            self.logger_info('train dataset has {} samples,{} in dataloader'.format(len(self.train_loader.dataset), self.train_loader_len))

    def _train_epoch(self, epoch):
        self.model.train()
        epoch_start = time.time()
        batch_start = time.time()
        train_loss = 0.
        running_metric_text = runningScore(2)
        lr = self.optimizer.param_groups[0]['lr']

        for i, batch in enumerate(self.train_loader):
            if i >= self.train_loader_len:
                break
            self.global_step += 1
            lr = self.optimizer.param_groups[0]['lr']

            # 数据进行转换和丢到gpu
            for key, value in batch.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
            cur_batch_size = batch['img'].size()[0]

            preds = self.model(batch['img'])
            loss_dict = self.criterion(preds, batch)
            # backward
            self.optimizer.zero_grad()
            loss_dict['loss'].backward()
            self.optimizer.step()
            if self.config['lr_scheduler']['type'] == 'WarmupPolyLR':
                self.scheduler.step()
            # acc iou
            score_shrink_map = cal_text_score(preds[:, 0, :, :], batch['shrink_map'], batch['shrink_mask'], running_metric_text,
                                              thred=self.config['post_processing']['args']['thresh'])

            # loss 和 acc 记录到日志
            loss_str = 'loss: {:.4f}, '.format(loss_dict['loss'].item())
            for idx, (key, value) in enumerate(loss_dict.items()):
                loss_dict[key] = value.item()
                if key == 'loss':
                    continue
                loss_str += '{}: {:.4f}'.format(key, loss_dict[key])
                if idx < len(loss_dict) - 1:
                    loss_str += ', '

            train_loss += loss_dict['loss']
            acc = score_shrink_map['Mean Acc']
            iou_shrink_map = score_shrink_map['Mean IoU']

            if self.global_step % self.log_iter == 0:
                batch_time = time.time() - batch_start
                self.logger_info(
                    '[{}/{}], [{}/{}], global_step: {}, speed: {:.1f} samples/sec, acc: {:.4f}, iou_shrink_map: {:.4f}, {}, lr:{:.6}, time:{:.2f}'.format(
                        epoch, self.epochs, i + 1, self.train_loader_len, self.global_step, self.log_iter * cur_batch_size / batch_time, acc,
                        iou_shrink_map, loss_str, lr, batch_time))
                batch_start = time.time()

            if self.tensorboard_enable and self.config['local_rank'] == 0:
                # write tensorboard
                for key, value in loss_dict.items():
                    self.writer.add_scalar('TRAIN/LOSS/{}'.format(key), value, self.global_step)
                self.writer.add_scalar('TRAIN/ACC_IOU/acc', acc, self.global_step)
                self.writer.add_scalar('TRAIN/ACC_IOU/iou_shrink_map', iou_shrink_map, self.global_step)
                self.writer.add_scalar('TRAIN/lr', lr, self.global_step)
                if self.global_step % self.show_images_iter == 0:
                    # show images on tensorboard
                    self.inverse_normalize(batch['img'])
                    self.writer.add_images('TRAIN/imgs', batch['img'], self.global_step)
                    # shrink_labels and threshold_labels
                    shrink_labels = batch['shrink_map']
                    threshold_labels = batch['threshold_map']
                    shrink_labels[shrink_labels <= 0.5] = 0
                    shrink_labels[shrink_labels > 0.5] = 1
                    show_label = torch.cat([shrink_labels, threshold_labels])
                    show_label = vutils.make_grid(show_label.unsqueeze(1), nrow=cur_batch_size, normalize=False, padding=20, pad_value=1)
                    self.writer.add_image('TRAIN/gt', show_label, self.global_step)
                    # model output
                    show_pred = []
                    for kk in range(preds.shape[1]):
                        show_pred.append(preds[:, kk, :, :])
                    show_pred = torch.cat(show_pred)
                    show_pred = vutils.make_grid(show_pred.unsqueeze(1), nrow=cur_batch_size, normalize=False, padding=20, pad_value=1)
                    self.writer.add_image('TRAIN/preds', show_pred, self.global_step)
        return {'train_loss': train_loss / self.train_loader_len, 'lr': lr, 'time': time.time() - epoch_start,
                'epoch': epoch}

    def _eval(self, epoch):
        self.model.eval()
        # torch.cuda.empty_cache()  # speed up evaluating after training finished
        raw_metrics = []
        total_frame = 0.0
        total_time = 0.0
        for i, batch in tqdm(enumerate(self.validate_loader), total=len(self.validate_loader), desc='test model'):
            with torch.no_grad():
                # 数据进行转换和丢到gpu
                for key, value in batch.items():
                    if value is not None:
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(self.device)
                start = time.time()
                preds = self.model(batch['img'])

                # 可视化预测图
                shrink_maps = preds[:, 0, :, :]
                threshold_maps = preds[:, 1, :, :]
                binary_maps = preds[:, 2, :, :]
                aa = np.array(shrink_maps.cpu().data.numpy()[0, ...] * 255, np.uint8)
                bb = np.array(threshold_maps.cpu().data.numpy()[0, ...] * 255, np.uint8)
                cc = np.array(binary_maps.cpu().data.numpy()[0, ...] * 255, np.uint8)
                # cc = np.array(shrink_maps.cpu().data.numpy()[0, ...]*threshold_maps.cpu().data.numpy()[0, ...]* 255, np.uint8)
                # cv2.imshow("img_aa", aa)
                # cv2.imshow("img_bb", bb)
                # cv2.imshow("img_cc", cc)
                # cv2.waitKey(10)

                boxes, scores = self.post_process(batch, preds,is_output_polygon=self.metric_cls.is_output_polygon)
                total_frame += batch['img'].size()[0]
                total_time += time.time() - start
                raw_metric = self.metric_cls.validate_measure(batch, (boxes, scores))
                raw_metrics.append(raw_metric)
        metrics = self.metric_cls.gather_measure(raw_metrics)
        self.logger_info('FPS:{}'.format(total_frame / total_time))
        return metrics['recall'].avg, metrics['precision'].avg, metrics['fmeasure'].avg

    def my_env_show(self,epoch):
        print(len(self.validate_loader))
        loss_all = [] # 记录损失
        for i, batch in tqdm(enumerate(self.validate_loader), total=len(self.validate_loader), desc='test model'):
            with torch.no_grad():
                # 数据进行转换和丢到gpu
                for key, value in batch.items():
                    if value is not None:
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(self.device)
                preds = self.model(batch['img'])
                # 可视化预测图
                shrink_maps = preds[:, 0, :, :]
                threshold_maps = preds[:, 1, :, :]
                binary_maps = preds[:, 2, :, :]
                # aa = np.array(shrink_maps.cpu().data.numpy()[0, ...] * 255, np.uint8)
                # bb = np.array(threshold_maps.cpu().data.numpy()[0, ...] * 255, np.uint8)
                cc = np.array(binary_maps.cpu().data.numpy()[0, ...] * 255, np.uint8)
                # cc = np.array(shrink_maps.cpu().data.numpy()[0, ...]*threshold_maps.cpu().data.numpy()[0, ...]* 255, np.uint8)
                # cv2.imshow("img_aa", aa)
                # cv2.imshow("img_bb", bb)
                # cv2.imshow("img_cc", cc)
                # cv2.waitKey(10)
                # if epoch % 20 == 0:
                #     cv2.imwrite(f"output/eval_img/{epoch}_{i}.jpg", cc)

                alpha = 1.0
                beta = 10
                ohem_ratio=3
                eps=1e-6
                bce_loss = BalanceCrossEntropyLoss(negative_ratio=ohem_ratio)
                dice_loss = DiceLoss(eps=eps)
                l1_loss = MaskL1Loss(eps=eps)
                try:
                    # 输入的数据进行转换
                    batch['img'] = batch['img'][0].permute(1,2,0) # 3,W,H -> W,H,3
                    batch['text_polys'] = batch['text_polys'][0]  # 去掉第一个维度
                    batch['ignore_tags'] = batch['ignore_tags'][0] # 去掉第一个维度
                    batch = self.MakeShrinkMap(batch)   # 构建Shrink图
                    batch = self.MakeBorderMap(batch)   # 构建Border图

                    # ======================= 测试输入的数据是否正确 =================== #
                    # ww = np.array(batch["threshold_map"]* 255, np.uint8)
                    # xx = np.array(batch["threshold_mask"]* 255, np.uint8)
                    # yy = np.array(batch['shrink_mask']* 255, np.uint8)
                    # zz = np.array(batch['shrink_map']* 255, np.uint8)
                    # cv2.imshow("ww",ww)
                    # cv2.imshow("xx",xx)
                    # cv2.imshow("yy", yy)
                    # cv2.imshow("zz", zz)
                    # cv2.waitKey(0)

                    # 将numpy数据转为torch格式
                    shrink_map,shrink_mask = torch.from_numpy(batch['shrink_map'][None]), torch.from_numpy(batch['shrink_mask'][None])
                    # 计算shrink图的二值交叉熵损失
                    loss_shrink_maps = bce_loss(shrink_maps.cuda(),shrink_map.cuda(),shrink_mask.cuda() )
                    # print(loss_shrink_maps)
                    # 将numpy数据转为torch格式
                    threshold_map,threshold_mask = torch.from_numpy(batch['threshold_map'][None]), torch.from_numpy(batch['threshold_mask'][None])
                    # 计算threshold图的l1损失
                    loss_threshold_maps = l1_loss(threshold_maps.cuda(), threshold_map.cuda(), threshold_mask.cuda())
                    # print(loss_threshold_maps)
                    if preds.size()[1] > 2:
                        # 将numpy数据转为torch格式
                        loss_binary_maps = dice_loss(binary_maps.cuda(), shrink_map.cuda(),shrink_mask.cuda())
                        loss_all.append(alpha * loss_shrink_maps + beta * loss_threshold_maps + loss_binary_maps)
                except Exception as e:
                    print(e)
        if len(loss_all)!=0:
            self.logger_info('epoch:{}, val_loss: {}'.format(epoch, sum(loss_all)/len(loss_all)))
            if sum(loss_all)/len(loss_all) < self.eval_save_loss:
                import shutil
                shutil.copy('{}/model_latest.pth'.format(self.checkpoint_dir), '{}/model_val_best.pth'.format(self.checkpoint_dir))
                self.eval_save_loss = sum(loss_all)/len(loss_all)

    def _on_epoch_finish(self):
        self.logger_info('[{}/{}], train_loss: {:.4f}, time: {:.4f}, lr: {}'.format(
            self.epoch_result['epoch'], self.epochs, self.epoch_result['train_loss'], self.epoch_result['time'],
            self.epoch_result['lr']))
        net_save_path = '{}/model_latest.pth'.format(self.checkpoint_dir)
        net_save_path_best = '{}/model_best.pth'.format(self.checkpoint_dir)

        if self.config['local_rank'] == 0:
            self._save_checkpoint(self.epoch_result['epoch'], net_save_path)
            save_best = False

            self.metric_cls = None# 不适用指标保存模型
            if self.metric_cls is None:
                self.my_env_show(self.epoch_result['epoch'])

            if self.validate_loader is not None and self.metric_cls is not None:  # 使用f1作为最优模型指标
                recall, precision, hmean = self._eval(self.epoch_result['epoch'])

                if self.tensorboard_enable:
                    self.writer.add_scalar('EVAL/recall', recall, self.global_step)
                    self.writer.add_scalar('EVAL/precision', precision, self.global_step)
                    self.writer.add_scalar('EVAL/hmean', hmean, self.global_step)
                self.logger_info('test: recall: {:.6f}, precision: {:.6f}, f1: {:.6f}'.format(recall, precision, hmean))

                if hmean >= self.metrics['hmean']:
                    save_best = True
                    self.metrics['train_loss'] = self.epoch_result['train_loss']
                    self.metrics['hmean'] = hmean
                    self.metrics['precision'] = precision
                    self.metrics['recall'] = recall
                    self.metrics['best_model_epoch'] = self.epoch_result['epoch']
            else:
                if self.epoch_result['train_loss'] <= self.metrics['train_loss']:
                    save_best = True
                    self.metrics['train_loss'] = self.epoch_result['train_loss']
                    self.metrics['best_model_epoch'] = self.epoch_result['epoch']
            best_str = 'current best, '
            for k, v in self.metrics.items():
                best_str += '{}: {:.6f}, '.format(k, v)
            if self.metric_cls is not None:
                self.logger_info(best_str)
            if save_best:
                import shutil
                shutil.copy(net_save_path, net_save_path_best)
                self.logger_info("Saving current best: {}".format(net_save_path_best))
            else:
                self.logger_info("Saving checkpoint: {}".format(net_save_path))


    def _on_train_finish(self):
        for k, v in self.metrics.items():
            self.logger_info('{}:{}'.format(k, v))
        self.logger_info('finish train')
