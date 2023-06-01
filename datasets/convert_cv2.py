import json
import cv2
import numpy as np

# 读取COCO格式的标注文件
with open('./result.json') as f:
    coco_data = json.load(f)

# 遍历每个图像
for img_data in coco_data['images']:
    # 获取图像信息
    img_id = img_data['id']
    img_name = img_data['file_name']
    img_width = img_data['width']
    img_height = img_data['height']

    # 创建空的二进制掩码
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # 获取该图像对应的所有标注信息
    ann_data = coco_data['annotations']
    ann_data = [ann for ann in ann_data if ann['image_id'] == img_id]

    # 遍历每个标注信息
    for ann in ann_data:
        # 获取分割掩码
        segment = ann['segmentation'][0]
        poly = np.array(segment).reshape((int(len(segment) / 2), 2)).astype(np.int32)
            # 将分割掩码转换为二进制掩码
        cv2.fillPoly(mask, [poly], color=1)

    # 保存掩码图像
    out_path = f"mask/{img_name[:-4]}.png"
    print(out_path)
    cv2.imwrite(out_path, mask)
