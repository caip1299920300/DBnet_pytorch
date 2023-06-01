import json
import math
# 读取COCO格式的标注文件
with open('result.json') as f:
    coco_data = json.load(f)

# 遍历每个图像
for img_data in coco_data['images']:
    # 获取图像信息
    img_id = img_data['id']
    img_name = img_data['file_name']
    img_width = img_data['width']
    img_height = img_data['height']

    # 获取该图像对应的所有标注信息
    ann_data = coco_data['annotations']
    ann_data = [ann for ann in ann_data if ann['image_id'] == img_id]

    txt_next = ""
    # 遍历每个标注信息
    for ann in ann_data:
        # 获取分割掩码
        segment = ann['segmentation'][0]
        if(len(segment)<6): # 跳过少于两个坐标的点
            continue
        for index,xy in enumerate(segment):
            if xy<0: xy=0
            if index%2==0:
                if xy >img_width: xy = img_width
            else:
                if xy >img_height: xy = img_height
            txt_next+=str(round(xy))+","
        txt_next+="DJB\n"
        # print(txt_next)
    
    with open(f'labels/{img_name.split("/")[-1].replace(".jpg",".txt")}','w') as file:
        file.write(txt_next)
    