import numpy as np
import random
import mnist_load

class network():
    """创建一个用于辨识手写数值0-9的三层神经网络"""

    def __init__(self, net):
        """net为包含每层neurons数量的list，例如[2,3,1]为三层神经网络
        2为输入层neuron数，3为隐藏层neuron数，1为输出层neuron数。获取
        神经网络层数以及初始随机权重偏置数值（用了高斯分布）"""
        self.cb = []
        self.cw = []
        self.net = net
        self.layers_num = len(net)  # 层数

        # 初始化每一层的神经元个数
        self.layer_neurons_num = []
        for i in range(self.layers_num):
            self.layer_neurons_num.append(net[i])

        # 通过高斯分布初始化层与层之间的权重
        self.weights = []
        # 为方便下标计算，加一层空的在前面
        self.weights.append([0])
        for i in range(1, self.layers_num):
            self.weights.append(np.random.randn(self.layer_neurons_num[i], self.layer_neurons_num[i-1]))

        # 通过高斯分布初始化层与层之间的偏置
        self.biases = []
        # 为方便下标计算，加一层空的在前面
        self.biases.append([0])
        for i in range(1, self.layers_num):
            self.biases.append(np.random.randn(self.layer_neurons_num[i], 1))

        # 初始化所有z，下标为0的行是不用的
        self.z = []
        for i in range(self.layers_num):
            self.z.append(np.zeros(self.layer_neurons_num[i]))

        # 初始化所有a,下标为0的行是不用的
        self.a = []
        for i in range(self.layers_num):
            self.a.append(np.zeros(self.layer_neurons_num[i]))

    # 小批量梯度下降法
    def MBGD(self, training_data, epochs, mini_batch_size, eta, testing_data=None):
        """training_data`` is a list containing 50,000 2-tuples ``(x, y)``.
        ``x`` is a 784-dimensional numpy.ndarray containing the input image.
        ``y`` is a 10-dimensional numpy.ndarray representing the unit vector
        corresponding to the correct digit for ``x``，epochs表迭代次数，
        mini_batch_size表小批量样本的样本数量，eta表示学习速率"""
        n = len(training_data)  # 获取训练集样本总数量
        for i in range(epochs):
            random.shuffle(training_data)  # 将序列training_data所有元素随机排序
            # 将training_data平均分割,n/mini_batch_size得到的是float，range只能是int，用//代替/即可得到整数
            mini_batches = [training_data[j:j + mini_batch_size] for j in range(n // mini_batch_size)]

            for a_batch_of_training_data in mini_batches:  # 取其中一组小批量数据利用反向传播算法更新一次权重偏置
                self.update_w_b(a_batch_of_training_data, eta)

            if testing_data:
                test_correct = 0  # test_correct表示测试正确的次数
                for (x, y) in testing_data:
                    a = self.feedForward(x)  # 得到某一测试样本输出层激活值
                    amax = np.argmax(a)  # 取最大输出激活值的索引
                    test_correct = test_correct + int(amax == y)  # 对测试正确的次数进行求和
                print('epoch', i, 'accuracy:', test_correct/len(testing_data))  # 输出每个epoch的测试正确次数/总测试次数（测试正确率）
            else:
                print('没有测试数据集')

    def feedForward(self, x):
        """输入输入层activation，返回输出层activation"""
        self.a[0] = x
        for i in range(1, self.layers_num):
            self.z[i] = np.dot(self.weights[i], self.a[i-1]) + self.biases[i]
            self.a[i] = self.sigmoid(self.z[i])
        # 返回输出，也就是最后一层
        return self.a[self.layers_num-1]

    def backPropagation(self, y):
        # 对最后一层处理
        # loss函数对a求导,loss函数是（1/2）*（y-a）^2
        partial_derivative_ca = (self.a[self.layers_num-1] - y)
        # 乘以a对z求导
        delta = partial_derivative_ca * self.derivative_delta(self.z[self.layers_num - 1])

        # 对其他层处理
        for i in range(self.layers_num-1, 0, -1):
            self.cw[i] = self.cw[i] + np.dot(delta, self.a[i-1].transpose())
            self.cb[i] = self.cb[i] + delta
            if i != 1:  # 不算z[0]那项
                delta = (np.dot((self.weights[i]).transpose(), delta) * self.derivative_delta(self.z[i-1]))

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def update_w_b(self, training_data, eta):
        """利用反向传播算法计算梯度并调整权重、偏置"""
        # 创建权重梯度的零矩阵，方便放置计算出的权重梯度
        self.cw = []
        # 加一行空方便计算下标
        self.cw.append([0])
        for i in range(1, len(self.weights)):
            self.cw.append(np.zeros(self.weights[i].shape))

        # 创建偏置梯度的零矩阵，方便放置计算出的偏置梯度
        self.cb = []
        # 加一行空方便计算下标
        self.cb.append([0])
        for i in range(1, len(self.biases)):
            self.cb.append(np.zeros(self.biases[i].shape))

        for x, y in training_data:
            # 前向传播
            self.feedForward(x)
            # 反向传播
            self.backPropagation(y)
        # 计算平均权重、偏置梯度，对权重、偏置进行一次调整
        for i in range(1, self.layers_num):
            self.weights[i] = self.weights[i] - eta * self.cw[i] / len(training_data)
            self.biases[i] = self.biases[i] - eta * self.cb[i] / len(training_data)

    def derivative_delta(self, z):
        """求误差的导数"""
        return self.sigmoid(z) * (1 - self.sigmoid(z))


if __name__ == '__main__':
    training_Data, validation_data, testing_data = mnist_load.load_data_wrapper()
    net = network([784, 100, 100, 10])
    net.MBGD(training_data=training_Data, epochs=50, mini_batch_size=10, eta=4.5, testing_data=testing_data)
