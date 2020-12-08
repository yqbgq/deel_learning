import os

# 屏蔽了 TF 的输出，不然会影响进度条的输出，有点麻烦
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

import tensorflow as tf
from tensorflow.keras import layers, optimizers, Sequential, datasets, metrics

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
    def __init__(self, batch_size, learning_rate, epochs, l_val):
        self.batch_size = batch_size        # 每个 batch 的大小
        self.lr = learning_rate             # 学习率
        self.epochs = epochs                # 学习多少个 epoch
        self.lambda_val = l_val             # 正则化中的 lambda

    def train(self):
        """学习函数"""
        self.__get_data()  # 获取数据
        self.__build_model()  # 构造模型
        self.__train()  # 进行训练

    def __get_data(self):
        """加载数据集"""
        (x, y), (x_test, y_test) = datasets.cifar10.load_data()

        y = tf.squeeze(y, axis=1)                       # 从 (N, 1) => (N)
        y_test = tf.squeeze(y_test, axis=1)

        # 将数据和标签组合成训练数据集合，随机打乱时的缓冲区域为1000，使用 preprocess 进行处理
        # 每个 batch 的大小为 128
        # 需要注意的是这里使用了 map 方法进行数据集处理，好像不用全部加载入显存，不会导致爆显存
        self.train_db = tf.data.Dataset.from_tensor_slices((x, y))
        self.train_db = self.train_db.shuffle(1000).map(preprocess).batch(128)

        self.test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        self.test_db = self.test_db.map(preprocess).batch(64)

    def __build_model(self):
        """构造模型"""
        self.model = Sequential([
            # 卷积单元 1
            layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

            # 卷积单元 2
            layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

            # 卷积单元 3
            layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

            # 卷积单元 4
            layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

            # 卷积单元 5
            layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

            # 将图像打平成为一维向量
            layers.Flatten(),

            # 全连接层
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(10, activation=None),
        ])

        # 构造模型，其输入为 N * 32 * 32 * 3
        self.model.build(input_shape=[None, 32, 32, 3])

    def __train(self):
        """训练的具体过程"""
        optimizer = optimizers.Adam(lr=self.lr)             # 使用 Adam 优化器
        acc_meter = metrics.Accuracy()                      # 定义准确率计算器

        for epoch in range(self.epochs):
            with trange(len(self.train_db)) as t:

                iter_data = iter(self.train_db)             # 将训练数据集转换为迭代器

                for step in t:                              # 使用 tqdm 优化输出
                    t.set_description(                      # 修改表头显示，增加时间和epoch显示
                        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) +
                        " Epoch {}".format(epoch)
                    )

                    (x, y) = next(iter_data)                # 获取一个训练的batch

                    with tf.GradientTape() as tape:         # 记录梯度
                        result = self.model(x)              # 通过网络得到的结果
                        y_label = tf.one_hot(y, depth=10)   # 转化获得对应的标签，用于计算准确率

                        # 计算准确率
                        acc_meter.update_state(tf.argmax(result, axis=1), tf.argmax(y_label, axis=1))

                        # 计算损失函数，使用交叉熵函数
                        loss = tf.losses.categorical_crossentropy(y_label, result, from_logits=True)
                        loss = tf.reduce_mean(loss)

                        # 计算正则化损失函数，叠加得到最后的损失
                        regularization_list = [tf.nn.l2_loss(par) for par in self.model.trainable_variables]
                        loss += self.lambda_val * tf.reduce_sum(tf.stack(regularization_list))

                    # 计算梯度，并且应用，进行 BP
                    grads = tape.gradient(loss, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                    # 规范化输出信息
                    info = "Step: {}, Loss: {:.4f}, ACC: {:.2f}% " \
                           "".format(step, float(loss), acc_meter.result().numpy() * 100)

                    # 在每个 epoch 的最后，计算在测试集上的 ACC
                    if step == len(self.train_db) - 1:
                        total_num = 0
                        total_correct = 0
                        for x, y in self.test_db:   # 计算正确预测的样本数量
                            result = self.model(x)

                            prob = tf.nn.softmax(result, axis=1)
                            pred = tf.argmax(prob, axis=1)
                            pred = tf.cast(pred, dtype=tf.int32)

                            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
                            correct = tf.reduce_sum(correct)

                            total_num += x.shape[0]
                            total_correct += int(correct)

                        acc = total_correct / total_num

                        info += "ACC in Test Data: {:.2f}%".format(acc * 100)

                    t.set_postfix_str(info)
                    if step % 100 == 0:
                        t.write("")         # 每执行 100 此迭代，进行换行


if __name__ == "__main__":
    # 构造 cifar 实例，每个 batch的大小为 128， 学习率大小为 0.001
    # 学习 20 个 epoch， 正则化函数的 lambda 为 0.0001
    c = cifar(128, 0.001, 50, 0.0001)
    c.train()
