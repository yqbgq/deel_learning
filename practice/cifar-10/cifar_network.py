import os

# 屏蔽了 TF 的输出，不然会影响进度条的输出，有点麻烦
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

import tensorflow as tf
from tensorflow.keras import layers, optimizers, Sequential, datasets

from tqdm import trange
import time

# 设置显存增长式分配，小显存机器太难了
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def preprocess(x, y):
    """
    对数据进行处理的函数，将图像处理到 -1 ~ 1之间

    :param x: 图像数据
    :param y: 标签
    :return: 处理好的图像数据和标签
    """
    # [0~1]
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)
    return x, y


class cifar:
    def __init__(self, batch_size, learning_rate, epochs):
        self.batch_size = batch_size    # 每个 batch 的大小
        self.lr = learning_rate         # 学习率
        self.epochs = epochs            # 学习多少个 epoch

    def train(self):
        """学习函数"""
        self.__get_data()               # 获取数据
        self.__build_model()            # 构造模型
        self.__train()                  # 进行训练

    def __get_data(self):
        (x, y), (x_test, y_test) = datasets.cifar10.load_data()
        y = tf.squeeze(y, axis=1)
        y_test = tf.squeeze(y_test, axis=1)

        self.train_db = tf.data.Dataset.from_tensor_slices((x, y))
        self.train_db = self.train_db.shuffle(1000).map(preprocess).batch(128)

        self.test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        self.test_db = self.test_db.map(preprocess).batch(64)

    def __build_model(self):
        self.model = Sequential([
            layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

            # unit 2
            layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

            # unit 3
            layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

            # unit 4
            layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

            layers.Flatten(),

            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(10, activation=None),
        ])

        self.model.build(input_shape=[None, 32, 32, 3])

    def __train(self):
        optimizer = optimizers.Adam(lr=self.lr)

        for epoch in range(self.epochs):
            with trange(len(self.train_db)) as t:

                iter_data = iter(self.train_db)

                for step in t:
                    t.set_description(
                        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) +
                        " Epoch {}".format(epoch)
                    )
                    (x, y) = next(iter_data)

                    with tf.GradientTape() as tape:
                        result = self.model(x)
                        y_label = tf.one_hot(y, depth=10)

                        loss = tf.losses.categorical_crossentropy(y_label, result, from_logits=True)
                        loss = tf.reduce_mean(loss)

                        regularization_list = [tf.nn.l2_loss(par) for par in self.model.trainable_variables]

                        loss += 0.0001 * tf.reduce_sum(tf.stack(regularization_list))

                    grads = tape.gradient(loss, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                    if step % 100 == 0:
                        t.set_postfix_str(str(epoch) + " " + str(step) + ' loss: ' + str(float(loss)))
                        t.write()
                    if step == len(self.train_db) - 1:

                        total_num = 0
                        total_correct = 0
                        for x, y in self.test_db:
                            logits = self.model(x)
                            prob = tf.nn.softmax(logits, axis=1)
                            pred = tf.argmax(prob, axis=1)
                            pred = tf.cast(pred, dtype=tf.int32)

                            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
                            correct = tf.reduce_sum(correct)

                            total_num += x.shape[0]
                            total_correct += int(correct)

                        acc = total_correct / total_num
                        t.set_postfix_str(str(epoch) + " " + str(step) + ' loss: ' + str(float(loss)) + str(
                            epoch) + " " + "acc: " + str(acc))


c = cifar(128, 0.001, 20)
c.train()
