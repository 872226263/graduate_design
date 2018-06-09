# coding: utf-8
# @Time    : 2018/4/5 1:20
# @Author  : ndsry
# @FileName: build_image_data
# @Software: PyCharm
# @Github    ：http://github.com/872226263
# ==========================================

import tensorflow as tf
import numpy as np
import time
import random
from PIL import Image
from divImg import get_names_list




# path of images.txt and labels.txt
IMAGES_TXT_PATH = './singleNumber/images.txt'
LABELS_TXT_PATH = './singleNumber/labels.txt'
TRAIN_FILE_PATH = './singleNumber'

IMAGE_WIDTH = 70
IMAGE_HIGHT = 30

CHAR_SET_LEN = 10
CAPTCHA_LEN = 4

TRAIN_IMAGE_PERCENT = 0.8

# path of model which had been trained
MODEL_SAVE_PATH = './models/'

# get img and covert to grey img
def get_data_and_label(fileName):
    img = Image.open(fileName)
    # convert to grey img
    img = img.convert('L')
    image_array = np.array(img)
    image_data = image_array.flatten()/255
    return image_data

def numbersToArray(numbers):
    label = np.zeros(CAPTCHA_LEN * CHAR_SET_LEN)
    for i,c in enumerate(numbers):
        idx = i * CHAR_SET_LEN + ord(c) - ord('0')
        label[idx] = 1
    return label


# 生成一个训练batch
def get_next_batch(batchSize=32, trainOrTest='train', step=0):
    batch_data = np.zeros([batchSize, IMAGE_WIDTH * IMAGE_HIGHT])
    batch_label = np.zeros([batchSize, CAPTCHA_LEN * CHAR_SET_LEN])
    fileNameList = TRAINING_IMAGE_NAME
    if trainOrTest == 'validate':
        fileNameList = VALIDATION_IMAGE_NAME

    totalNumber = len(fileNameList)
    indexStart = step * batchSize
    for i in range(batchSize):
        index = (i + indexStart) % totalNumber
        name = fileNameList[index]
        img_data = get_data_and_label(name)
        img_label = numbersToArray(name[-8:-4])
        batch_data[i, :] = img_data
        batch_label[i, :] = img_label

    return batch_data, batch_label

def train_data_with_CNN():
    # 初始化权值
    def weight_variable(shape, name='weight'):
        init = tf.truncated_normal(shape, stddev=0.1)
        var = tf.Variable(initial_value=init, name=name)
        return var
        # 初始化偏置

    def bias_variable(shape, name='bias'):
        init = tf.constant(0.1, shape=shape)
        var = tf.Variable(init, name=name)
        return var

        # 卷积
    def conv2d(x, W, name='conv2d'):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)

        # 池化
    def max_pool_2X2(x, name='maxpool'):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            # 计算参数的均值，并使用tf.summary.scaler记录
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)

            # 计算参数的标准差
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            # 用直方图记录参数的分布
            tf.summary.histogram('histogram', var)

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32,[None,IMAGE_HIGHT*IMAGE_WIDTH],name='data-input')
        y = tf.placeholder(tf.float32,[None,CAPTCHA_LEN*CHAR_SET_LEN],name='label-input')
        with tf.name_scope('input_reshape'):
            x_input = tf.reshape(x,[-1,IMAGE_HIGHT,IMAGE_WIDTH,1],name='x-input')
            tf.summary.image('x-input', x_input, 1)

    # 请注意 keep_prob 的 name，在测试model时会用到它
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32, name='keep-prob')
        tf.summary.scalar('dropout_keep_probability', keep_prob)

    # 第一层卷积
    with tf.name_scope('Conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32], 'W_conv1')
        B_conv1 = bias_variable([32], 'B_conv1')

        variable_summaries(W_conv1)
        variable_summaries(B_conv1)

        conv1 = tf.nn.relu(conv2d(x_input, W_conv1, 'conv1') + B_conv1)

        # for i in range(32):
        #     conv1_summary = tf.reshape(conv1[:][:][:][i], [-1, 30, 70, 1], name='conv1-summary')
        #     tf.summary.image('conv1_summary', conv1_summary, 1)

    with tf.name_scope('Pool1'):
        conv1 = max_pool_2X2(conv1, 'conv1-pool')
        conv1 = tf.nn.dropout(conv1, keep_prob)

    # 第二层卷积
    with tf.name_scope('Conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64], 'W_conv2')
        B_conv2 = bias_variable([64], 'B_conv2')
        conv2 = tf.nn.relu(conv2d(conv1, W_conv2, 'conv2') + B_conv2)

    with tf.name_scope('Pool2'):
        conv2 = max_pool_2X2(conv2, 'conv2-pool')
        conv2 = tf.nn.dropout(conv2, keep_prob)

    # 第三层卷积
    with tf.name_scope('Conv3'):
        W_conv3 = weight_variable([5, 5, 64, 64], 'W_conv3')
        B_conv3 = bias_variable([64], 'B_conv3')
        conv3 = tf.nn.relu(conv2d(conv2, W_conv3, 'conv3') + B_conv3)
    with tf.name_scope('Pool3'):
        conv3 = max_pool_2X2(conv3, 'conv3-pool')
        conv3 = tf.nn.dropout(conv3, keep_prob)
    # 全链接层
    # 每次池化后，图片的宽度和高度均缩小为原来的一半，进过上面的三次池化，宽度和高度均缩小8倍
    with tf.name_scope('dense'):
        W_fc1 = weight_variable([9 * 4 * 64, 1024], 'W_dense')
        B_fc1 = bias_variable([1024], 'B_dense')
        fc1 = tf.reshape(conv3, [-1, 9 * 4 * 64])
        fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, W_fc1), B_fc1))
        fc1 = tf.nn.dropout(fc1, keep_prob)
    # 输出层
    with tf.name_scope('output'):
        W_fc2 = weight_variable([1024, CAPTCHA_LEN * CHAR_SET_LEN], 'W_fc2')
        B_fc2 = bias_variable([CAPTCHA_LEN * CHAR_SET_LEN], 'B_fc2')
        output = tf.add(tf.matmul(fc1, W_fc2), B_fc2, 'output')
        output_image = tf.reshape(output, [-1, 4, 10, 1], name='output-image')
        tf.summary.image('output-image', output_image, 1)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output))
        optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
        tf.summary.scalar('loss', loss)

    predict = tf.reshape(output, [-1, CAPTCHA_LEN, CHAR_SET_LEN], name='predict')
    labels = tf.reshape(y, [-1, CAPTCHA_LEN, CHAR_SET_LEN], name='labels')

    predict2 = tf.reshape(output, [-1, CAPTCHA_LEN, CHAR_SET_LEN,1], name='predict')
    #tf.summary.image('a', predict2[0], 3)

    # 预测结果
    # 请注意 predict_max_idx 的 name，在测试model时会用到它
    predict_max_idx = tf.argmax(predict, axis=2, name='predict_max_idx')
    labels_max_idx = tf.argmax(labels, axis=2, name='labels_max_idx')
    predict_correct_vec = tf.equal(predict_max_idx, labels_max_idx)
    accuracy = tf.reduce_mean(tf.cast(predict_correct_vec, tf.float32))
    #记录accurarcy
    tf.summary.scalar('accuracy', accuracy)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # summaries合并
        merged = tf.summary.merge_all()
        # 写到指定的磁盘路径中
        train_writer = tf.summary.FileWriter('./log/train', sess.graph)
        #test_writer = tf.summary.FileWriter('./log/test', sess.graph)
        #初始化
        sess.run(tf.global_variables_initializer())
        steps = 0
        for epoch in range(6000):
            train_data, train_label = get_next_batch(64, 'train', steps)
            _,summary_str = sess.run([optimizer,merged], feed_dict={x: train_data, y: train_label, keep_prob: 0.75})
            train_writer.add_summary(summary_str, steps)
            if steps % 100 == 0:

                test_data, test_label = get_next_batch(64, 'validate', steps)
                acc = sess.run(accuracy, feed_dict={x: test_data, y: test_label, keep_prob: 1.0})
                #test_writer.add_summary(summary_str, steps)
                print("steps=%d, accuracy=%f" % (steps, acc))
                if acc > 0.999:
                    saver.save(sess, MODEL_SAVE_PATH + "crack_captcha.model", global_step=steps)
                    break
            steps += 1

if __name__ == '__main__':
    image_filename_list = get_names_list('./src_img')
    total = len(image_filename_list)
    random.seed(time.time())
    #打乱顺序
    random.shuffle(image_filename_list)
    trainImageNumber = int(total * TRAIN_IMAGE_PERCENT)
    #分成测试集
    TRAINING_IMAGE_NAME = image_filename_list[ : trainImageNumber]
    #和验证集
    VALIDATION_IMAGE_NAME = image_filename_list[trainImageNumber : ]
    train_data_with_CNN()
    print('Training finished')