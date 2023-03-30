import os
import os.path
import sys
import random
import numpy as np
import logging
import argparse
import torch
import torchvision
import pytorch_ssim
from torch.utils.tensorboard import SummaryWriter

import global_v as glv
from network_parser import parse
from datasets import load_dataset_snn

from utils import *
import fsvae_models.fsvae as fsvae
from fsvae_models.snn_layers import LIFSpike
from scipy.spatial.distance import cdist
from itertools import combinations
from dmsvdd import kmeans_plus_plus, update_c, update_r, get_c_min_max, KMeansPlusPlus, distance

# 定义前置超参数
init_device = torch.device("cuda:0")
device = init_device
max_accuracy = 0
min_loss = 1000
nu = 0.1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--checkpoint', action='store', dest='checkpoint',
                        help='The path of checkpoint, if use checkpoint')
    parser.add_argument('--config', action='store', dest='config', help='The path of config file')
    parser.add_argument('--device', type=int)

    # 读取一些信息，并新建一些文件夹
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(0)

    if args.config is None:
        raise Exception('Unrecognized config file.')

    if args.device is None:
        init_device = torch.device("cuda:0")
    else:
        init_device = torch.device(f"cuda:{args.device}")

    os.makedirs(f'checkpoint/{args.name}', exist_ok=True)
    writer = SummaryWriter(log_dir=f'checkpoint/{args.name}/tb')
    logging.basicConfig(filename=f'checkpoint/{args.name}/{args.name}.log', level=logging.INFO)

    logging.info("start parsing settings")

    params = parse(args.config)
    network_config = params['Network']

    logging.info("finish parsing settings")
    logging.info(network_config)
    print(network_config)

    glv.init(network_config, [args.device])

    dataset_name = glv.network_config['dataset']
    data_path = glv.network_config['data_path']

    logging.info("dataset loading...")

    # 数据集加载
    train_loader_id_list,test_loader_id_list = load_data_all(glv.network_config['dataset'],1)
    test_loader_ood_list = load_data_all(glv.network_config['ood_data'],0)
    logging.info("dataset loaded")

    # 加载现有模型_待完成
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        checkpoint = torch.load(checkpoint_path)
        if "Letters" in checkpoint_path:
            # 加载网络
            if network_config['model'] == 'FSVAE':
                net = fsvae.FSVAE(class_num=26)
            elif network_config['model'] == 'FSVAE_large':
                net = fsvae.FSVAELarge()
            else:
                raise Exception('not defined model')

            net = net.to(init_device)
            net.load_state_dict(checkpoint['net'])
            R = checkpoint['R']
            c = checkpoint['c']
            print("n_class = 26")
            print('R = ',R)
            print('c = ',torch.mean(c))
        else:
            # 加载网络
            if network_config['model'] == 'FSVAE':
                net = fsvae.FSVAE(class_num=10)
            elif network_config['model'] == 'FSVAE_large':
                net = fsvae.FSVAELarge()
            else:
                raise Exception('not defined model')

            net = net.to(init_device)
            net.load_state_dict(checkpoint['net'])
            R = checkpoint['R']
            c = checkpoint['c']
            print("n_class = 10")
            print('R = ',R)
            print('c = ',torch.mean(c))

        # 开始测试异常检测性能
        print('test start!')
        auc_list = []
        aupr_list = []
        if 'KMNIST' in checkpoint_path:
            train_loader_id_list,test_loader_id_list = load_data_all(['KMNIST'],1)
            test_loader = test_loader_id_list[0]
            dataset_name = 'KMNIST'
        elif 'Letters' in checkpoint_path:
            train_loader_id_list,test_loader_id_list = load_data_all(['Letters'],1)
            test_loader = test_loader_id_list[0]
            dataset_name = 'Letters'
        elif 'MNIST' in checkpoint_path:
            train_loader_id_list,test_loader_id_list = load_data_all(['MNIST'],1)
            test_loader = test_loader_id_list[0]
            dataset_name = 'MNIST'
        elif 'FashionMNIST' in checkpoint_path:
            train_loader_id_list,test_loader_id_list = load_data_all(['FashionMNIST'],1)
            test_loader = test_loader_id_list[0]
            dataset_name = 'FashionMNIST'
            
        for _ood, test_loader_ood in zip(glv.network_config['ood_data'], test_loader_ood_list):
            if _ood != dataset_name:
                print("ood-data = ", _ood)
                anoscores_in = get_anoscore(net, test_loader, c,network_config,R)
                anoscores_ood = get_anoscore(net, test_loader_ood, c,network_config,R)
                labels_test = [0] * len(anoscores_in) + [1] * len(anoscores_ood)
                scores = anoscores_in + anoscores_ood
                auc, aupr = auc_and_aupr(labels_test, scores)
                auc_list.append(round(auc, 3))
                aupr_list.append(round(aupr, 3))
        print("auc_list = ", auc_list)
        print("aupr_list = ", aupr_list)
        sys.exit()

    best_loss = 1e8
    latent_dim = glv.network_config['latent_dim']

    for train_data, test_data, id_data_name,class_num in \
            zip(train_loader_id_list,test_loader_id_list,network_config['dataset'],network_config['class_num']):
        # 加载网络
        if network_config['model'] == 'FSVAE':
            net = fsvae.FSVAE(class_num)
        elif network_config['model'] == 'FSVAE_large':
            net = fsvae.FSVAELarge()
        else:
            raise Exception('not defined model')

        net = net.to(init_device)

        optimizer = torch.optim.AdamW(net.parameters(),
                                  lr=glv.network_config['lr'],
                                  betas=(0.9, 0.999),
                                  weight_decay=0.001)
        
        print("id_data = ", id_data_name)

        for seed in network_config['seed']:
            print("seed = ",seed)
            setup_seed(seed)

            # 随机初始化R和c
            c = torch.randn(size=(class_num, latent_dim), device=device)
            net.update_c(c)
            R = torch.ones(size=(class_num,), device=device) * 5
            for e in range(glv.network_config['epochs']):
                write_weight_hist(net, e,writer)
                if network_config['scheduled']:
                    net.update_p(e, glv.network_config['epochs'])
                    logging.info("update p")
                latent_dim = glv.network_config['latent_dim']

                if e<network_config['pre_epochs']:
                    train_loss = train(net, train_data, optimizer, c, R, e,glv,writer,network_config,name=args.name,class_num=class_num)
                    #test_loss = test(net, test_loader, e,ood=0,c=c,R=R,writer=writer)
                    #test_loss_ood = test(net,test_loader_ood_list[4],e,ood=1,c=c,R=R,writer=writer)
                elif e == network_config['pre_epochs']:
                    print("epoch reaches the threshold, update c and R !")
                    # 更新R和c，其中c只更新一次
                    c = update_c(device, train_data, net, class_num, seed)
                    net.update_c(c)
                    R = update_r(device, train_data, net, c, nu,class_num)
                    print("Update R and C successfully!")

                    train_loss = train(net, train_data, optimizer, c, R, e, glv,writer,network_config,name=args.name,class_num=class_num)
                    #test_loss = test(net, test_loader, e, ood=0, c=c, R=R,writer=writer)
                    #test_loss_ood = test(net, test_loader_ood_list[4], e, ood=1, c=c, R=R,writer=writer)
                elif e>network_config['pre_epochs']:
                    # 仅在外面更新R
                    #c = update_c(device, train_data, net, class_num, seed)
                    if e%10 == 0:
                        R = update_r(device, train_data, net, c, nu,class_num)
                    train_loss = train(net, train_data, optimizer, c, R, e, glv,writer,network_config,name=args.name,class_num=class_num)
                    #test_loss = test(net, test_loader, e, ood=0, c=c, R=R)
                    #test_loss_ood = test(net, test_loader_ood_list[4], e, ood=1, c=c, R=R)

                # 保存模型,R和c
                if e == glv.network_config['epochs']-1:
                    R = update_r(device, train_data, net, c, nu,class_num)
                net_params = {'net':net.state_dict(),'R':R,'c':c}
                torch.save(net_params, f'checkpoint/{args.name}/checkpoint'+'_'+id_data_name+'_'+str(seed)+'.pth')
                #if test_loss < best_loss:
                #   best_loss = test_loss
                #   torch.save(net.state_dict(), f'checkpoint/{args.name}/best.pth')

                # 采样模型
                #sample(net, e, batch_size=128)

            # 开始测试异常检测性能
            print('test start!')
            get_auc_and_aupr_list(network_config['ood_data'],test_loader_ood_list,c,id_data_name,net,test_data,network_config,writer,R)
