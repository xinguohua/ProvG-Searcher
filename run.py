from common import utils
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from subgraph_matching import train
import argparse
import helper.create_candidate as cn
import helper.create_pos_neg_dict as cd

from subgraph_matching import config 
import os

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Order embedding arguments')
    utils.parse_optimizer(parser)
    config.parse_encoder(parser)
    args = parser.parse_args()
    # 创建负采样数据和正负样本字典的函数
    sampling_stats = f'data/{args.data_identifier}/test_neg_dict_{args.numberOfNeighK}.pc'
    if not os.path.exists(sampling_stats):
        print('first sampling stats will be created')
        print('this is a one time process for each dataset')
        cn.run(args.numberOfNeighK,args.data_identifier)
        cd.run(args.data_identifier,args.numberOfNeighK) 
    # 加载数据集
    ### load dataset and set utils parameters accordingly
    utils.loadDatas(feature=args.data_identifier, numberOfNeighK=args.numberOfNeighK, args=args)
    utils.numberOfFeature = len(list(utils.abstractType2array.values())[0])
    args.feature_size = utils.numberOfFeature + 17
    utils.glob_feature=args.data_identifier
    # 启动训练过程
    train.train_loop(args)