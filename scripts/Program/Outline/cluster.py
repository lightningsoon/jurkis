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
workdir='./parameters/'
def load_data():
    imlist = glob.glob('/Users/huanghao/Desktop/data/*.png')[:]

    # imgs=np.array(list(image.img_to_array(image.load_img(i,target_size=(224,224))) for i in imlist))
    imgs=np.array(list(image.img_to_array(image.load_img(i,target_size=(224,224))) for i in imlist))
    imnbr = len(imlist)  # get the number of images
    print("The number of images is %d" % imnbr)
    # top_model = MobileNet(weights='imagenet', include_top=False)
    #从本地
    # imgs.astype(np.uint8)
    # top_model.save('mobilenet.h5')
    # exit()
    # cv2.imshow('1', imgs[0].astype(np.uint8))
    # cv2.waitKey(0)
    # exit()
    return imgs

#%%
# def kmeans(data,K=4):
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#     compactness,labels,centers = cv2.kmeans(data,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
#     return compactness,labels,centers
#%%
# def cvtRGB(img):
#     t1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     t2=cv2.resize(t1,(20,20))
#     return t2
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
            # if kwargs:
            #     plt.figure(1,figsize=(3,8))
            #     plt.subplot(L+K,1,i+1+i_sum)
            #     plt.plot(kwargs['hists'][ind[i]])
            #     plt.axis('off')
        i_sum+=len(ind)+1


def downDimension(data,train_model=False,n_components = 18):
    '''
    主成分分析,降维
    :param data:
    :return:
    '''
    file_name=workdir+'pca'+str(n_components)+'.m'
    if train_model or not os.path.isfile(file_name):
        pca = PCA(n_components=n_components)
        pca.fit(data)
        joblib.dump(pca,file_name)
        print('pca(n=%d) explained variance ratio: %.4f' % (n_components,np.sum(pca.explained_variance_ratio_)))
        #('pca18 ratio:', 0.9724457)
    else:
        pca=joblib.load(file_name)
    results = pca.transform(data)
    return results
#%%
# cv2.PCAProject()
# 提取特征（颜色，形状）
# model=cv2.xfeatures2d.SURF_create()
# plt.figure()
# def get_Mask(im):
#     #im hsv
#     # 手动掩膜
#     im_raw=im.copy()
#     assert im.shape[0]>=70 and im.shape[1]>=40
#     x,y=np.array(im.shape[:2])//2
#     y=y+3
#     shift=5
#     h_mean=np.mean(im[y-shift:y+shift,x-shift:x+shift,0])
#     h_mean=(max(h_mean-20,0),min(h_mean+20,180))
#     mask_range=[(h_mean[0],0,100),(h_mean[1],255,255)]
#     mask=cv2.inRange(im,mask_range[0],mask_range[1])
#     result=cv2.bitwise_and(im_raw,im_raw,mask=mask)
#     return result
# def get_Mask2(im_bgr):
#     # 聚类掩膜
#     Z = im_bgr.reshape((-1, 3))
#
#     # convert to np.float32
#     Z = np.float32(Z)
#
#     # define criteria, number of clusters(K) and apply kmeans()
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#     K = 4
#     ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
#
#     # Now convert back into uint8, and make original image
#     center = np.uint8(center)
#     res = center[label.flatten()]
#     result = res.reshape((im_bgr.shape))
#     return result
# def get_Hist(imgs):
#     hists = []
#     for img in imgs:
#         # pro_img=get_Mask2(img)
#         img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         # pro_img=get_Mask(img_hsv)
#         hist = cv2.calcHist([img_hsv], [0], None, [180], [0, 180])[:]
#         # plt.plot(hist)
#         # plt.show()
#         hists.append(hist)
#     return hists

def get_x(ims):
    imgs=ims.copy()
    # print(imgs.shape)#(21, 224, 224, 3)
    imgs=preprocess_input(imgs)#归一化
    # print(imgs.shape)#(21, 224, 224, 3)
    t1=time.time()
    features=top_model.predict(imgs)
    t2=time.time()
    # results.shape=(-1,7,7,1024)
    print('per picture speed time(ms):',(t2-t1)/len(features))#0.2 CPU
    features=np.reshape(features,(len(features),-1))
    return features
def save_centroid_npz(centroid,K,update_npz=False,filename=workdir+'sample.npz'):
    #  保存 centers，K
    if update_npz or not os.path.isfile(filename) :
        np.savez(filename,centroid,K)
    else:
        npdata = np.load(filename)
        centroid, K = list(npdata[na] for na in npdata.files)
        centroid = np.array(centroid,dtype=np.float32)
    return centroid,K
# plot clusters
# for k in range(4):
#     ind = where(code==k)[0]
#     figure()
#     gray()
#     for i in range(minimum(len(ind),40)):
#         subplot(4,10,i+1)
#         imshow(immatrix[ind[i]].reshape((25,25)))
#         axis('off')
# show()
#%%
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
    if max_Score<th[max_Score_idx]:
        max_Score_idx=5
    return max_Score_idx
def two_D_visualization(datas,labels,cents):
    data2d = datas[:, :2]
    print(data2d.shape)
    for k in range(max(labels+1)):
        idx=np.where(labels == k)[0]
        plt.scatter(data2d[idx, 1], data2d[idx, 0],label=str(k),s=20)
        plt.scatter(cents[k][1],cents[k][0],marker='+',s=30)


def get_nearest_neighbor_threshold(datas,cents,labels):
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
        return dist_threshold
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
    np.save(file_name,dist_threshold)
    return dist_threshold
if __name__ == '__main__':
    K = 4
    # hists=get_Hist(imgs)
    filename=workdir+'data.npy'
    imgs = load_data()
    if  not os.path.isfile(filename) :
        top_model = load_model(workdir+'mobilenet.h5', custom_objects={
            'relu6': mobilenet.relu6,
            'DepthwiseConv2D': mobilenet.DepthwiseConv2D})
        datas = get_x(imgs)
        datas = downDimension(datas)
        np.save(filename, datas)
    else:
        datas = np.load(filename)

    print(datas.shape)
    # exit()
    centroid, labels, inertia = k_means(np.float32(datas), K,copy_x=False)
    centroid, K=save_centroid_npz(centroid,K,True)

    neighbor_threshold=get_nearest_neighbor_threshold(datas,centroid,labels)
    new_labels=[]
    for i in range(len(datas)):
        new_labels.append(nearest_neighbor(datas[i],centroid,neighbor_threshold))
    two_D_visualization(datas[:,:2],labels,centroid[:,:2])
    draw_into_a_pic(K, labels, imgs)
    print(np.array(new_labels))
    print(labels)
    plt.show()