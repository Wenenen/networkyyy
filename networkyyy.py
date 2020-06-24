import numpy as np
import random



class network():
    '''创建一个用于辨识手写数值0-9的三层神经网络'''
    def __init__(self,net):
        '''net为包含每层neurons数量的list，例如[2,3,1]为三层神经网络
        2为输入层neuron数，3为隐藏层neuron数，1为输出层neuron数。获取
        神经网络层数以及初始随机权重偏置数值（用了高斯分布）'''
        self.net = net
        self.num_layers=len(net)#层数
        self.firstlayer_neu=net[0]#第一层的神经元数量
        self.secondlayer_neu=net[1]#第二层的神经元数量
        self.thirdlayer_neu=net[2]#第三层的神经元数量
        self.weights=[np.random.randn(self.secondlayer_neu,self.firstlayer_neu)]#通过高斯分布获得第一、二层神经网络之间权重的随机初始值
        self.weights.append(np.random.randn(self.thirdlayer_neu,self.secondlayer_neu))#追加第二、三层神经网络之间权重的初始值
        self.biases =[np.random.randn(self.secondlayer_neu,1)]#获得第一、二层神经网络之间偏置的随机初始值
        self.biases.append(np.random.randn(self.thirdlayer_neu,1))#追加第二、三层神经网络之间偏置的初始值
        
    def MBGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
        '''training_data`` is a list containing 50,000 2-tuples ``(x, y)``.  
        ``x`` is a 784-dimensional numpy.ndarray containing the input image. 
        ``y`` is a 10-dimensional numpy.ndarray representing the unit vector 
        corresponding to the correct digit for ``x``，epochs表迭代次数，
        mini_batch_size表小批量样本的样本数量，eta表示学习速率'''
        n=len(training_data)#获取训练集样本总数量
        for i in range(epochs):
            random.shuffle(training_data)#将序列training_data所有元素随机排序
            mini_batches=[training_data[j:j + mini_batch_size] for j in range(n//mini_batch_size)]#将training_data平均分割,n/mini_batch_size得到的是float，range只能是int，用//代替/即可得到整数
            
            for mini_batch in mini_batches:#取其中一组小批量数据利用反向传播算法更新一次权重偏置
                self.update_w_b(mini_batch,eta)
                
            if test_data:
                test_correct=0#test_correct表示测试正确的次数
                m=len(test_data)#获取测试集样本总数量
                for (x,y) in test_data:
                    a=self.feedword(x)#得到某一测试样本输出层激活值
                    amax=np.argmax(a)#取最大输出激活值的索引
                    test_correct=test_correct+int(amax==y)#对测试正确的次数进行求和
                print('Eopch',i,':',test_correct,'/',m)#输出每个epoch的测试正确次数/总测试次数（测试正确率）
            else:
                print('没有测试数据集')
                    
    def feedword(self,x):
        '''输入输入层activation，返回输出层activation'''
        a1=self.sigmoid(np.dot(self.weights[0],x)+self.biases[0])
        a2=self.sigmoid(np.dot(self.weights[1],a1)+self.biases[1])
        return a2 
    
    def sigmoid(self, z):
        a=1.0/(1.0+np.exp(-z))
        return a
    
    def update_w_b(self,mini_batch,eta):  
        '''利用反向传播算法计算梯度并调整权重、偏置'''
        #创建偏置梯度的零矩阵，方便放置计算出的偏置梯度
        Cb = [np.zeros(self.biases[0].shape)]
        Cb.append(np.zeros(self.biases[1].shape))
        #创建权重梯度的零矩阵，方便放置计算出的权重梯度
        Cw = [np.zeros(self.weights[0].shape)]
        Cw.append(np.zeros(self.weights[1].shape))
        for x,y in mini_batch:
            #将输入值进行前向传播得到每层的activation
            z1=np.dot(self.weights[0],x)+self.biases[0]#输入第二层的带权输入
            a1=self.sigmoid(z1)#第二层的activation
            z2=np.dot(self.weights[1],a1)+self.biases[1]#输入第三层（输出层）的带权输入
            a2=self.sigmoid(z2)#第三层（输出层）的activation
            #计算输出层误差
            Part_derivative_Ca=(a2-y)#代价函数对aL的偏导数
            delta2=Part_derivative_Ca*self.derivative_delta(z2)#利用Hadamard乘积计算输出层误差
            #反向传播
            delta1=(np.dot((self.weights[1]).transpose(),delta2)*self.derivative_delta(z1))#求取倒数第二层误差
            Cw[-1]=Cw[-1]+np.dot(delta2, a1.transpose())#得到第二层和第三层之间的权重梯度
            Cb[-1]=Cb[-1]+delta2#得到第二层和第三层之间的偏置梯度
            Cw[-2]=Cw[-2]+np.dot(delta1, x.transpose())#得到第一层和第二层之间的权重梯度
            Cb[-2]=Cb[-2]+delta1#得到第一层和第二层之间的偏置梯度
        #计算平均权重、偏置梯度，对权重、偏置进行一次调整
        # print(Cb[-1]/10.0)
        self.weights[-1]=self.weights[-1]-eta*Cw[-1]/len(mini_batch)
        self.weights[-2]=self.weights[-2]-eta*Cw[-2]/len(mini_batch)
        self.biases[-1]=self.biases[-1]-eta*Cb[-1]/len(mini_batch)
        self.biases[-2]=self.biases[-2]-eta*Cb[-2]/len(mini_batch)    
    
    def derivative_delta(self,z): 
        '''求误差的倒数'''
        return self.sigmoid(z)*(1-self.sigmoid(z))  
    
import mnist_load
training_data, validation_data, test_data = mnist_load.load_data_wrapper()

net =network([784,100,10])
net.MBGD(training_data,30,10,6.5,test_data=test_data)           
        
    
                
                
                
                
            
            
    
        
