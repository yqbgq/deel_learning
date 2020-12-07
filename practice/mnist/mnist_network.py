"""
最简单的入门 Hello World！！
MNIST 手写数字集识别

"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, datasets, metrics

from tqdm import trange


class mnist:
    def __init__(self, opt="1"):
        self.opt = opt
        self.__get_data()
        self.__build_model()

    def train(self):
        if self.opt == "1":
            self.__train_compile_fit()
        elif self.opt == "2":
            self.__train_manual()

    def __get_data(self):
        (self.train_x, self.train_y), (self.test_x, self.test_y) = datasets.mnist.load_data()
        self.train_x = tf.convert_to_tensor(self.train_x, dtype=tf.float32) / 255.
        self.test_x = tf.convert_to_tensor(self.test_x, dtype=tf.float32) / 255.
        if self.opt == "2":
            self.train_y = tf.one_hot(self.train_y, depth=10)
            self.test_y = tf.one_hot(self.test_y, depth=10)

    def __build_model(self):
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10)
        ])
        self.model.build(input_shape=(None, 28 * 28))

    def __train_compile_fit(self):
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        self.model.fit(self.train_x, self.train_y, epochs=10)
        test_loss, test_acc = self.model.evaluate(self.test_x, self.test_y, verbose=1)
        print('\nTest accuracy:', test_acc)

    def __train_manual(self):
        optimizer = optimizers.Adam(learning_rate=0.001)
        acc_meter = metrics.Accuracy()
        for epoch in range(10):
            with trange(1000, desc="Epoch {}: ".format(epoch)) as t:
                for step in t:
                    with tf.GradientTape() as tape:
                        train_x = tf.reshape(self.train_x, (-1, 28 * 28))
                        out = self.model(train_x)
                        loss = tf.reduce_mean(tf.square(out - self.train_y))
                        acc_meter.update_state(tf.argmax(out, axis=1), tf.argmax(self.train_y, axis=1))
                        info_part_1 = "Loss: {:.4f} Acc:{:.2f}% ".format(loss, acc_meter.result().numpy() * 100)
                        t.set_postfix_str(info_part_1)

                    grads = tape.gradient(loss, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                    if step == 999:
                        test_acc_meter = metrics.Accuracy()
                        out = self.model(self.test_x)
                        loss = tf.reduce_mean(tf.square(out - self.test_y))
                        test_acc_meter.update_state(tf.argmax(out, axis=1), tf.argmax(self.test_y, axis=1))

                        info_part_2 = "In test dataset, Loss: {:.4f}, Acc: {:.2f}%".format(float(loss),
                                                                                           test_acc_meter.result().numpy() * 100)
                        t.set_postfix_str(info_part_1 + info_part_2)


mnist_net = mnist(opt="2")
mnist_net.train()
