"""
创建 ： 2020-12-7

最简单的入门 Hello World！！ MNIST 手写数字集识别 在这里实现了两种学习方式：1.模型编译以及拟合方式 2.手动训练方式
前者更加方便，且输出看起来比较舒服，后者更加灵活，输出需要自己调整
"""

import os

# 屏蔽了 TF 的输出，不然会影响进度条的输出，有点麻烦
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, datasets, metrics

from tqdm import trange


class mnist:
    def __init__(self, opt="1"):
        """
        MNIST 初始化函数

        :param opt: 表示使用的训练方法，时使用编译拟合方式还是手动训练方式
        """
        self.opt = opt
        self.__get_data()       # 获取数据
        self.__build_model()    # 建立模型

    def train(self):
        if self.opt == "1":
            self.__train_compile_fit()
        elif self.opt == "2":
            self.__train_manual()

    def __get_data(self):
        (self.train_x, self.train_y), (self.test_x, self.test_y) = datasets.mnist.load_data()
        self.train_x = tf.convert_to_tensor(self.train_x, dtype=tf.float32) / 255.  # 将数值预处理到 0-1 之间
        self.test_x = tf.convert_to_tensor(self.test_x, dtype=tf.float32) / 255.
        if self.opt == "2":
            self.train_y = tf.one_hot(self.train_y, depth=10)   # 如果是手动训练方式，可以首先转换成 one_hot 方式
            self.test_y = tf.one_hot(self.test_y, depth=10)     # 便于后续使用

    def __build_model(self):
        self.model = keras.Sequential([
            keras.layers.Flatten(),                             # 将输入的图片矩阵打平
            keras.layers.Dense(128, activation='relu'),         # 全连接层，输出128维特征
            keras.layers.Dense(10)                              # 全连接层，输出10维特征，表示属于每个数字的概率
        ])
        self.model.build(input_shape=(None, 28 * 28))           # 建立模型，设置输入数据的维度

    def __train_compile_fit(self):
        # 编译模型，使用 Adam 优化器，使用交叉熵损失函数，使用准确度进行评价
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        # 进行拟合，使用 10 个 Epoch
        self.model.fit(self.train_x, self.train_y, epochs=10)
        # 在测试集上进行评价
        test_loss, test_acc = self.model.evaluate(self.test_x, self.test_y, verbose=1)
        print('\nTest accuracy:', test_acc)

    def __train_manual(self):
        # 手动训练
        optimizer = optimizers.Adam(learning_rate=0.001)    # 使用 Adam 优化器，恒定学习率
        acc_meter = metrics.Accuracy()                      # 使用准确率评价器
        epochs = 10                                         # 使用 10 个Epoch
        steps = 500                                         # 每个 Epoch 训练 500 步

        for epoch in range(epochs):
            with trange(steps, desc="Epoch {}: ".format(epoch)) as t:
                for step in t:
                    with tf.GradientTape() as tape:

                        train_x = tf.reshape(self.train_x, (-1, 28 * 28))           # 将输入的图像转换为一维向量
                        out = self.model(train_x)                                   # 获得经过模型的输出
                        loss = tf.reduce_mean(tf.square(out - self.train_y))        # 使用简单的 MSE 损失函数

                        # 计算准确度
                        acc_meter.update_state(tf.argmax(out, axis=1), tf.argmax(self.train_y, axis=1))
                        # 更新准确度和损失信息
                        info_part_1 = "Loss: {:.4f} Acc:{:.2f}% ".format(loss, acc_meter.result().numpy() * 100)
                        # 更新输出，这里使用了进度条输出，更加好看一点
                        t.set_postfix_str(info_part_1)

                    # 计算梯度
                    grads = tape.gradient(loss, self.model.trainable_variables)
                    # 进行梯度下降
                    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                    # 每个 Epoch 的最后在测试集上进行测试
                    if step == steps - 1:

                        # 获取经过神经网络的输出
                        out = self.model(self.test_x)
                        loss = tf.reduce_mean(tf.square(out - self.test_y))

                        # 计算准确度
                        test_acc_meter = metrics.Accuracy()
                        test_acc_meter.update_state(tf.argmax(out, axis=1), tf.argmax(self.test_y, axis=1))

                        # 计算输出信息
                        info_part_2 = "In test dataset, Loss: {:.4f}," \
                                      " Acc: {:.2f}%".format(float(loss),
                                                             test_acc_meter.result().numpy() * 100)
                        # 更新输出信息
                        t.set_postfix_str(info_part_1 + info_part_2)


mnist_net = mnist(opt="2")
mnist_net.train()
