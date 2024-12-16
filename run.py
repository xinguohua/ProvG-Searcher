import matplotlib.pyplot as plt
from common import utils
plt.rcParams.update({'font.size': 16})
from subgraph_matching import train
import argparse
from subgraph_matching import config

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Order embedding arguments')
    utils.parse_optimizer(parser)
    config.parse_encoder(parser)
    args = parser.parse_args()
    # 加载数据集
    utils.load_datas(feature=args.data_identifier, numberOfNeighK=args.numberOfNeighK, args=args)
    utils.prepare_feature(feature=args.data_identifier)
    utils.numberOfFeature = len(list(utils.abstractType2array.values())[0])
    args.feature_size = utils.numberOfFeature + 17
    utils.glob_feature=args.data_identifier
    # 启动训练过程
    train.train_loop(args)