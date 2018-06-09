# coding: utf-8
# @Time    : 2018/4/12 1:56
# @Author  : ndsry
# @Software: PyCharm
# @Github  :http://github.com/872226263
# ==========================================
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import tool.download_jpg as download_jpg

CAPTCHA_LEN = 4

MODEL_SAVE_PATH = './models/'
TEST_IMAGE_PATH = './test_img/'


def get_image_data_and_name(fileName, filePath=TEST_IMAGE_PATH):
    pathName = os.path.join(filePath, fileName)
    img = Image.open(pathName)
    # 转为灰度图
    img = img.convert("L")
    image_array = np.array(img)
    image_data = image_array.flatten() / 255
    image_name = fileName[0:CAPTCHA_LEN]
    return image_data, image_name

def get_image_data_from_image():
    src_img = download_jpg.retImgFromInternet()
    # 转为灰度图
    img = src_img.convert("L")
    image_array = np.array(img)
    image_data = image_array.flatten() / 255
    return image_data,src_img


def digitalStr2Array(digitalStr):
    digitalList = []
    for c in digitalStr:
        digitalList.append(ord(c) - ord('0'))
    return np.array(digitalList)


def model_test():
    nameList = []
    for pathName in os.listdir(TEST_IMAGE_PATH):
        nameList.append(pathName.split('/')[-1])
    totalNumber = len(nameList)
    # 加载graph
    saver = tf.train.import_meta_graph(MODEL_SAVE_PATH + "crack_captcha.model-500.meta")
    graph = tf.get_default_graph()
    # 从graph取得 tensor，他们的name是在构建graph时定义的(查看上面第2步里的代码)
    input_holder = graph.get_tensor_by_name("input/data-input:0")
    keep_prob_holder = graph.get_tensor_by_name("dropout/keep-prob:0")
    predict_max_idx = graph.get_tensor_by_name("predict_max_idx:0")
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_SAVE_PATH))
        count = 0
        for fileName in nameList:
            img_data, img_name = get_image_data_and_name(fileName, TEST_IMAGE_PATH)
            predict = sess.run(predict_max_idx, feed_dict={input_holder: [img_data], keep_prob_holder: 1.0})
            filePathName = TEST_IMAGE_PATH + fileName
            print(filePathName)
            img = Image.open(filePathName)
            predictValue = np.squeeze(predict)
            rightValue = digitalStr2Array(img_name)
            if np.array_equal(predictValue, rightValue):
                result = '正确'
                count += 1
            else:
                result = '错误'
            print('实际值：{}， 预测值：{}，测试结果：{}'.format(rightValue, predictValue, result))
            print('\n')

        print('正确率：%.2f%%(%d/%d)' % (count * 100 / totalNumber, count, totalNumber))

def model_test_online():
    # 加载graph
    saver = tf.train.import_meta_graph(MODEL_SAVE_PATH + "crack_captcha.model-500.meta")
    graph = tf.get_default_graph()
    # 从graph取得 tensor，他们的name是在构建graph时定义的(查看上面第2步里的代码)
    input_holder = graph.get_tensor_by_name("input/data-input:0")
    keep_prob_holder = graph.get_tensor_by_name("dropout/keep-prob:0")
    predict_max_idx = graph.get_tensor_by_name("predict_max_idx:0")
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_SAVE_PATH))
        correct = 0
        error = 0
        for i in range(1000):
            img_data,src_img = get_image_data_from_image()
            predict = sess.run(predict_max_idx, feed_dict={input_holder: [img_data], keep_prob_holder: 1.0})
            predictValue = np.squeeze(predict).tobytes()
            predictS = str(predictValue[0])+str(predictValue[8])+str(predictValue[16])+str(predictValue[24])
            result = download_jpg.upload_get_result(predictS)
            if result=='验证码正确':
                src_img.save('./result/'+predictS+'.jpg')
                correct += 1
            elif result=='验证码错误':
                error += 1
            print('预测值：{}，测试结果：{}'.format(predictS, result))

        print('正确率：%.2f%%(%d/%d)' % ((correct * 100+0.1) / (correct + error), correct, correct + error))


if __name__ == '__main__':
    model_test_online()