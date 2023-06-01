import os,sys
import random
# 该路径已经添加到系统的环境变量
sys.path.append(os.getcwd())
img_list = os.listdir("train/img")
img_path = "./datasets/train/img/"
gt_path = "./datasets/train/gt/"

test_path = random.sample(img_list,int(len(img_list)*0.1))
# 训练数据地址
train_ = ""
# 测试数据地址
test_ = ""

for i in img_list:
    if i not in test_path:
        train_ += img_path+i+"\t"+gt_path+i.replace(".jpg",".txt")+"\n"
    else:
        test_ += img_path+i+"\t"+gt_path+i.replace(".jpg",".txt")+"\n"

# 写入txt
with open("train.txt","w") as file:
    file.write(train_)
with open("test.txt","w") as file:
    file.write(test_)