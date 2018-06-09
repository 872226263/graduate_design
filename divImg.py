# coding: utf-8
# @Time    : 2018/4/11 12:04
# @Author  : ndsry
# @Software: PyCharm
# @Github  :http://github.com/872226263
# ==========================================
from PIL import Image

import string
import random
import os

def cutImg(imgName):
    im = Image.open(imgName)
    imArray = []
    for i in range(4):
        divIm = im.crop((5+15*i,2,18+15*i,28))
        divImgName = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
        divIm.save('./singleNumber/'+divImgName+'.jpg')
        with open('./singleNumber/images.txt','a') as f:
            f.write(divImgName+'\n')
        with open('./singleNumber/labels.txt','a') as f:
            f.write(imgName[-8:-4][i]+'\n')

def get_names_list(imgPath):
    filename_list = os.listdir(imgPath)
    for i in range(len(filename_list)):
        filename_list[i] = imgPath + '/' + filename_list[i]
    return filename_list

# for name in get_names_list('./src_img'):
#     cutImg(name)