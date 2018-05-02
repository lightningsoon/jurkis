# coding=utf-8
from __future__ import division
from matplotlib import pyplot as plt
from matplotlib import markers
import glob
import numpy as np
from keras.models import load_model
from keras.applications import MobileNet,mobilenet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing import image
from keras import backend as K
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.cluster import k_means
# 获取图像列表
import cv2
import os
import time
import json
workdir='/home/momo/Project/jurkis_ws/src/jurvis/scripts/Program/Outline/parameters/'
def restore_model():
    global top_model
    top_model = load_model(workdir+'mobilenet.h5', custom_objects={
                'relu6': mobilenet.relu6,
                'DepthwiseConv2D': mobilenet.DepthwiseConv2D})
    print('load cluster model')
def load_data():
    imlist = glob.glob('/home/momo/Project/jurkis_ws/src/jurvis/scripts/Program/Outline/data/*.png')[:]
    imgs=np.array(list(image.img_to_array(image.load_img(i,target_size=(224,224))) for i in imlist))
    #导入的图像已经成float32，不清楚
    imnbr = len(imlist)
    print("The number of images is %d" % imnbr)
    return imgs


def draw_into_a_pic(K,labels,imgs,**kwargs):
    # imgs=list(map(cvtRGB,imgs))
    L=len(imgs)
    imgs=imgs.astype(np.uint8)
    # plt.figure(figsize=(8,8))
    # fig, axes=plt.subplots(2,K//2 if K%2==0 else K//2+1)
    i_sum=0
    for k in range(K):
        ind = np.where(labels == k)[0]
        for i in range(len(ind)):
            plt.figure(0,figsize=(3,8))
            plt.subplot(L+K,1,i+1+i_sum)
            plt.imshow(imgs[ind[i]])
            plt.axis('off')
        i_sum+=len(ind)+1


def downDimension(data,train_model=False,n_components = 20):
    '''
    主成分分析,降维
    :param data:
    :return:
    '''
    file_name=workdir+'pca_'+str(n_components)+'.m'
    if train_model or not os.path.isfile(file_name):
        pca = PCA(n_components=n_components)
        pca.fit(data)
        joblib.dump(pca,file_name)
        # print('pca(n=%d) explained variance ratio: %.4f' % (n_components,np.sum(pca.explained_variance_ratio_)))
        #('pca18 ratio:', 0.9724457)
    else:
        pca=joblib.load(file_name)
    results = pca.transform(data)
    print('pca(n=%d) explained variance ratio: %.4f' % (n_components, np.sum(pca.explained_variance_ratio_)))
    return results


def get_x(ims):
    imgs=ims.copy()
    # print(imgs.shape)#(21, 224, 224, 3)
    imgs=preprocess_input(imgs)#归一化
    # print(imgs.shape)#(21, 224, 224, 3)
    # t1=time.time()
    features=top_model.predict(imgs)
    # t2=time.time()
    # results.shape=(-1,7,7,1024)
    # print('per picture speed time(ms):',(t2-t1)/len(features))#0.2 CPU #0.0469 GTX 960M
    features=np.reshape(features,(len(features),-1))
    return features
def save_centroid_npz(centroid,K,filename=workdir+'*means.npz',update_npz=False):
    #  保存 centers，K
    # 如果已经保存就恢复
    filename=glob.glob(filename)[0]
    if update_npz or not os.path.isfile(filename):
        np.savez(filename,centroid,K)
    else:
        npdata = np.load(filename)
        centroid, K = list(npdata[na] for na in npdata.files)
        centroid = np.array(centroid,dtype=np.float32)
    return centroid,K

def get_distance_score(x,cent):
    # 0～1，越大越近
    return 1/(1+np.linalg.norm(x-cent,axis=1))
def nearest_neighbor(x,cents,th=None):
    '''
    :param x: (1,D)
    :param cents: list(np.ndarray) (K,D)
    :th :判断是不是已知旧类的阈值
    :return:
    '''
    dists=get_distance_score(x,cents)
    max_Score_idx=np.argmax(dists)
    max_Score=dists[max_Score_idx]
    if np.all(th!=None):
        if max_Score<th[max_Score_idx]*0.8:#手动放宽范围
            max_Score_idx=5
    return max_Score_idx
def two_D_visualization(datas,labels,cents):
    data2d = datas[:, :2]
    print(data2d.shape)
    for k in range(max(labels+1)):
        idx=np.where(labels == k)[0]
        plt.scatter(data2d[idx, 1], data2d[idx, 0],label=str(k),s=40)
        plt.scatter(cents[k][1],cents[k][0],marker='+',s=50)


def get_nearest_neighbor_threshold(datas=None,cents=None,labels=None):
    '''
    得到阈值，这个是分数阈值（0，1）越小越远，所以要倒着排
    :param datas:
    :param cents:
    :param labels:
    :return:
    '''
    file_name=workdir+'Dist_ThresholdK='+str(len(cents))+'.npy'
    if os.path.isfile(file_name):
        dist_threshold=np.load(file_name)
    else:
        dist_threshold=[]
        for i,c in enumerate(cents):
            idx=np.where(labels==i)
            dists=get_distance_score(datas[idx],c)
            dists=sorted(dists,reverse=True)
            #2西格玛
            id=len(dists)-1
            # id= int(0.8*id if id >= 12 else id)
            dist_threshold.append(dists[id])
        # print(dist_threshold)
        dist_threshold=np.array(dist_threshold)
        np.save(file_name,dist_threshold)
    return dist_threshold
class Who_am_I(object):
    '''
    此类给ros使用，综合上面各个函数
    '''
    def __init__(self):
        restore_model()
        self.__cents,self.__K=save_centroid_npz(None,None)
        # print(self.__K)
        self.__get_x=get_x
        self.__down_Dimension=downDimension
        self.__th=get_nearest_neighbor_threshold(None,self.__K)
    def get_kind(self,img):
        img=np.resize(img,(224,224))
        x=self.__get_x(img[np.newaxis,:,:,:])
        x=self.__down_Dimension(x)
        label=nearest_neighbor(x, self.__cents, self.__th)
        return label


if __name__ == '__main__':
    K = 4
    # hists=get_Hist(imgs)
    filename=workdir+'data.npy'
    imgs = load_data()
    if  not os.path.isfile(filename) :
        restore_model()
        datas = get_x(imgs)
        datas = downDimension(datas)
        np.save(filename, datas)
    else:
        datas = np.load(filename)

    print(datas.shape)
    # exit()
    centroid, labels, inertia = k_means(np.float32(datas), K,copy_x=False)
    # 多试几次，肯定能有个好点初始值
    centroid, K=save_centroid_npz(centroid,K,False,filename=workdir+str(K)+'means.npz')

    neighbor_threshold=get_nearest_neighbor_threshold(datas,centroid,labels)
    new_labels=[]
    for i in range(len(datas)):
        new_labels.append(nearest_neighbor(datas[i],centroid,neighbor_threshold))
    two_D_visualization(datas[:,:2],labels,centroid[:,:2])
    draw_into_a_pic(K, labels, imgs)
    print(np.array(new_labels))
    print(labels)
    plt.show()