import random
import sys

import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.transforms import Lambda
from torchvision.datasets.vision import VisionDataset
from scipy.spatial.distance import cdist
from itertools import combinations
n_steps = 16
# Kmeans 类
class KMeansPlusPlus:
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X):
        # 初始化聚类中心点
        #print(X.shape)
        centers = kmeans_plus_plus(X,n_clusters=self.n_clusters,
                                   random_state=self.random_state)

        # 迭代更新聚类中心点和标签
        for _ in range(self.max_iter):
            # 计算每个点到每个中心点的距离
            distances = cdist(X, centers)

            # 更新标签
            labels = np.argmin(distances,axis=1)

            # 更新聚类中心点
            new_centers=np.array([np.mean(X[labels==i],axis=0) for i in range(self.n_clusters)])

            # 检查收敛条件 
            if np.allclose(centers,new_centers): 
              break 

            centers=new_centers 

        self.cluster_centers_=centers 
        self.labels_=labels 

def kmeans_plus_plus(X,n_clusters=8,random_state=None): 
    if random_state is not None: 
        np.random.seed(random_state) 

    # 选择第一个中心点 
    centers=[X[np.random.randint(0,len(X))]] 

    # 选择剩余的中心点 
    for _ in range(n_clusters-1): 
        # 计算每个点到最近中心点的距离 
        distances=np.min(cdist(X,centers),axis=1) 

        # 按照概率分布选择下一个中心点 
        probabilities=distances/np.sum(distances) 
        next_center_index=np.random.choice(len(X),p=probabilities) 
        centers.append(X[next_center_index]) 

    return np.array(centers)

# 获取中心C
# 仅使用1000个样本用于计算c
def update_c_v1(device, train_loader, net, class_num, seed):

    net.eval()
    with torch.no_grad():
        s = 0
        for batch_idx, (data, targets) in enumerate(train_loader):
            real_img = data.to(device, non_blocking=True)
            labels = targets.to(device, non_blocking=True)

            # direct spike input
            spike_input = real_img.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps)  # (N,C,H,W,T)
            _,_,_,_,outputs = net(spike_input)
            try:
                complete_outputs = torch.cat((complete_outputs, outputs), dim=0)
            except UnboundLocalError:
                complete_outputs = outputs
            if s>=50000:
                break
            else:
                s+=1
    complete_outputs = complete_outputs.detach().cpu().numpy()
    # <class 'numpy.ndarray'> (1024, 128)
    kmeans = KMeansPlusPlus(n_clusters= class_num, random_state= seed)
    kmeans.fit(complete_outputs)
    c = torch.from_numpy(kmeans.cluster_centers_).to(device)
    # print("c ",c.shape) [class_num, latent_dim]
    return c

# 获取半径R
def update_r_v1(device, train_loader, net, c, nu,class_num):
    net.eval()
    with torch.no_grad():
        s = 0
        for batch_idx, (data, targets) in enumerate(train_loader):
            real_img = data.to(device, non_blocking=True)
            labels = targets.to(device, non_blocking=True)

            # direct spike input
            spike_input = real_img.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps)  # (N,C,H,W,T)
            _, _, _, _, outputs = net(spike_input)
            # outputs与10个c比较，计算距离，取最近的C作为中心，需要得到标签
            dist = torch.cdist(outputs, c) # (batch_size, class_num)
            min_center_index = dist.argmin(dim=1) # (batch_size,)
            min_dist = torch.gather(dist, 1, min_center_index.unsqueeze(1)).squeeze(1) # (batch_size,)
            if s>=500000:
                break
            else:
                s+=1
    net.train()

    all_min_dist =  torch.zeros(class_num,).to(device)
    all_min_dist.index_add_(0, min_center_index, min_dist)
    #print(all_min_dist)

    radius = torch.zeros(class_num,)
    for i in range(0,class_num):
        radius[i] = torch.quantile(torch.sqrt(all_min_dist[i]), 1 - nu)

    return radius

def get_c_min_max(c):
    distances = list(map(lambda x: distance(*x), combinations(c,2)))
    max_distance = max(distances)
    min_distance = min(distances)   
    return min_distance, max_distance

def distance(p1,p2):
    return torch.dist(p1,p2)

def update_c(device, train_loader, net, class_num, seed):
    net.eval()
    latent_dim = 128
    c = torch.randn(size=(class_num, latent_dim), device=device)
    # 创建一个字典，用于存储各个类别的特征向量和计数器
    features = {i: {"sum": torch.zeros(latent_dim).to(device), "count": 0} for i in range(class_num)}
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(train_loader):
            real_img = data.to(device, non_blocking=True)
            labels = targets.to(device, non_blocking=True)
            spike_input = real_img.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps)  # (N,C,H,W,T)
            _, _, _, _, outputs,_ = net(spike_input)
            # outputs形状[batch_size，latent_dim]
            for i in range(len(targets)):
                label = targets[i].item()
                features[label]["sum"] += outputs[i]
                features[label]["count"] += 1
        c = torch.stack([features[i]["sum"] / features[i]["count"] for i in range(class_num)], dim=0)
    #print(c.shape)
    #sys.exit()
    return c

def update_r(device, train_loader, net, c, nu,class_num):
    net.eval()
    dist_list = []
    center_index_list = []
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(train_loader):
            real_img = data.to(device, non_blocking=True)
            labels = targets.to(device, non_blocking=True)

            # direct spike input
            spike_input = real_img.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps)  # (N,C,H,W,T)
            _, _, _, _, outputs,clf = net(spike_input)
            dist = torch.cdist(outputs, c)  # (batch_size, class_num),而clf的形状也为batch_size,class_num,

            one_hot = torch.zeros((labels.shape[0], class_num)).cuda()
            one_hot.scatter_(1, labels.unsqueeze(1), 1) # batch_size,class_num\

            dist = torch.mul(one_hot,dist)

            center_index = dist.argmax(dim=1)  # (batch_size,)

            dist = torch.gather(dist, 1, center_index.unsqueeze(1)).squeeze(1)  # (batch_size,)

            dist_list.append(dist)
            center_index_list.append(center_index)

    net.train()

    dist_tensor = torch.cat(dist_list)
    center_index_tensor = torch.cat(center_index_list)

    all_dist = torch.zeros(class_num, ).to(device)
    all_dist.index_add_(0, center_index_tensor, dist_tensor)

    count = torch.bincount(center_index_tensor)

    radius = torch.zeros(class_num, )
    for i in range(0, class_num):
        radius[i] = torch.quantile((all_dist[i]/count[i]), 1 - nu)
    return radius