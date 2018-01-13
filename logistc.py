# -*-coding:utf-8-*-
#!/usr/bin/python3
EPOCH=500
DECAYRATE=0.9
DECAY_STEP=100
LR=0.01
import numpy as np
import matplotlib.pyplot as plt
import math

def CreateData():
    featArr=np.array([[0.697,0.460],[0.774,0.376],[0.634,0.264], #数据来源于西瓜书
                      [0.608,0.318],[0.556,0.215],[0.403,0.237],
                      [0.481,0.149],[0.437,0.211],[0.666,0.091],
                      [0.243,0.267],[0.245,0.057],[0.343,0.099],
                      [0.639,0.161],[0.657,0.198],[0.360,0.370],
                      [0.593,0.042],[0.719,0.103]],dtype=np.float64)
    x0=np.ones((17,1))
    featArr=np.concatenate((featArr,x0),axis=1)
    labelArr=np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],dtype=np.int32)
    labelArr=labelArr[:,np.newaxis]
    return featArr,labelArr

# def Logistic(featArr,labelArr):
#     featMat=np.mat(featArr);labelMat=np.mat(labelArr)
#     lr=0.1
#     m,n=featMat.shape
#     paramArr=np.zeros((n,1))
#     for time in np.arange(10000):
#         for i in np.arange(m):
#             z=featMat[i].dot(paramArr)
#             h=1/(1+np.exp(-z))
#             for j in np.arange(n):
#                 paramArr[j]+=lr*(labelMat[i,0]-h[0,0])*featMat[i,j]
#     return paramArr

def Sigmoid(x):
    y=1/(1+np.exp(-x))
    return y
#批梯度上升法
def GradDescent(featArr,labelArr):
    featMat=np.mat(featArr);labelMat=np.mat(labelArr)
    m,n=featMat.shape
    lr=0.01
    paraArr=np.ones((n,1))
    for i in np.arange(EPOCH):
        z=featMat.dot(paraArr)
        y_pred=Sigmoid(z)
        error=labelMat-y_pred
        # for j in np.arange(n):
        paraArr+=lr*featMat.transpose().dot(error)
    return paraArr

#随机梯度上升法
def StocGradDescent(featArr,labelArr):
    featMat=np.mat(featArr);labelMat=np.mat(labelArr)
    m,n=featMat.shape
    lr=0.01
    weights=np.ones((n,1))
    for i in np.arange(EPOCH):
        for j in np.arange(m):
            z=featMat[j]*weights
            y_pred=Sigmoid(z)
            error=labelMat[j]-y_pred
            weights+=lr*featMat[j].transpose()*error
    return weights

#改进的随机梯度上升法
def StocGradDescent1(featArr,labelArr):
    lr=0.01
    featMat=np.mat(featArr);labelMat=np.mat(labelArr)
    m,n=np.shape(featMat)
    weights=np.ones((n,1))
    for i in np.arange(EPOCH):
        index=np.arange(m)
        for j in np.arange(m):
            lr=LR*pow(DECAYRATE,(j+i*m)/DECAY_STEP)
            choosedIndex=int(np.random.uniform(0,len(index)))
            item=index[choosedIndex]
            z=featMat[item]*weights
            y_pred=Sigmoid(z)
            error=labelMat[item]-y_pred
            weights+=lr*featMat[item].transpose()*error
            np.delete(index,choosedIndex)
    return weights



if __name__=='__main__':
    featArr,labelArr=CreateData()
    weights=StocGradDescent1(featArr,labelArr)
    weights1=StocGradDescent(featArr,labelArr)
    weights2=GradDescent(featArr,labelArr)
    # weights=weights.reshape()
    # z=np.dot(featArr,paraArr)
    # h=1/(1+np.exp(-z))
    # h[h>0.5]=1
    # h[h<=0.5]=0
    # pred=h
    # print(pred,labelArr)
    fig=plt.figure()
    ax1=fig.add_subplot(1,1,1)
    plt.scatter(featArr[0:8,0],featArr[0:8,1],s=30,c='r')
    plt.scatter(featArr[8:,0],featArr[8:,1],s=30,c='b')
    x=np.arange(0,1,0.01)
    plt.plot(x,(-weights[2]+weights[0]*x)/weights[1],'r--')
    plt.plot(x, (-weights1[2] + weights1[0] * x) / weights1[1],'b')
    plt.plot(x, (-weights2[2] + weights2[0] * x) / weights2[1],'g:')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()






