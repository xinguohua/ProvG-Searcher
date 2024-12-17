"""Train the order embedding model"""
from common.models import build_model

# Set this flag to True to use hyperparameter optimization
# We use Testtube for hyperparameter tuning
HYPERPARAM_SEARCH = False
HYPERPARAM_SEARCH_N_TRIALS = None   # how many grid search trials to run
                                    #    (set to None for exhaustive search)
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from common import data
from common import models
from common import utils
if HYPERPARAM_SEARCH:
    from subgraph_matching.hyp_search import parse_encoder
else:
    from subgraph_matching.config import parse_encoder
from subgraph_matching.test import validation



def make_data_source(args,isTrain=True):
    data_source = data.DiskDataSource(args.dataset, args.data_identifier,args.numberOfNeighK,
                node_anchored=args.node_anchored, feature_type=args.feature_type)
    return data_source


def train(args, model, data_source, scheduler, opt, clf_opt):
    """Train the order embedding model.
    args: Commandline arguments
    logger: logger for logging progress
    in_queue: input queue to an intersection computation worker
    out_queue: output queue to an intersection computation worker
    """
    loaders = data_source.gen_data_loaders(args.batch_size, train=True)
    for origin_batch in loaders:
        model.emb_model.train()
        model.emb_model.zero_grad()
        batch = data_source.gen_batch(origin_batch, train=True)
        batch = [elem.to(utils.get_device()) for elem in batch]
        pos_a, pos_b, neg_a, neg_b = batch
        # 遍历 batch 中的每个元素并打印其信息
        #     for i, b in enumerate(batch):
        #         print(f"Batch {i}:")
        #         print(f"  图的数量: {len(b)}")
        #         print(f"G 的值是: {b.G}, 图的数量是: {len(b.G)}")
        #         print(f"  节点特征形状: {b.node_feature.shape} 第一个节点特征: {b.node_feature[0]}")
        #         print(f"  节点标签索引形状: {b.node_label_index.shape} 第一个节点标签: {b.node_label_index[0]}")
        #         print(f"  边索引形状: {b.edge_index.shape}")
        #         print(f"  起点节点索引（第一行）: {b.edge_label_index[0]}")
        #         print(f"  终点节点索引（第二行）: {b.edge_label_index[1]}")
        #         print(f"  边类型数量: {len(b.type_edge)}  边类型: {b.type_edge}")
        #         print("-" * 50)
        emb_pos_a, emb_pos_b = model.emb_model(pos_a), model.emb_model(pos_b)
        emb_neg_a, emb_neg_b = model.emb_model(neg_a), model.emb_model(neg_b)

        emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
        emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)
        labels = torch.tensor([1] * pos_a.num_graphs + [0] * neg_a.num_graphs).to(
            utils.get_device())

        pred = model(emb_as, emb_bs)
        loss = model.criterion(pred, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if scheduler:
            scheduler.step()

        if args.method_type == "order":
            model.emb_model.eval()
            with torch.no_grad():
                pred = model.predict(pred)
            model.clf_model.zero_grad()
            pred = model.clf_model(torch.sigmoid(pred.unsqueeze(1)))
            criterion = nn.NLLLoss()
            clf_loss = criterion(pred, labels)
            clf_loss.backward()
            clf_opt.step()

        pred = pred.argmax(dim=-1)
        acc = torch.mean((pred == labels).type(torch.float))

        train_loss = loss.item()
        train_acc = acc.item()

        yield train_loss, train_acc
        

def train_loop(args):
    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))
    if not os.path.exists("plots/"):
        os.makedirs("plots/")

    print("Using dataset {}".format(args.dataset))
    record_keys = ["conv_type", "n_layers", "hidden_dim",
        "margin", "dataset", "max_graph_size", "skip"]
    args_str = ".".join(["{}={}".format(k, v)
        for k, v in sorted(vars(args).items()) if k in record_keys])
    logger = SummaryWriter(comment=args_str)
    model = build_model(args)
    if args.method_type == "order":
        clf_opt = optim.Adam(model.clf_model.parameters(), lr=0.01)
    else:
        clf_opt = None

    #############test#################
    print('create test points')
    data_source = make_data_source(args,isTrain=False)
    loaders = data_source.gen_data_loaders(args.batch_size, False)
    test_pts = []
    for origin_batch in loaders:
        pos_a, pos_b, neg_a, neg_b = data_source.gen_batch(origin_batch, False)
        if pos_a:
            pos_a = pos_a.to(torch.device("cpu"))
            pos_b = pos_b.to(torch.device("cpu"))
        neg_a = neg_a.to(torch.device("cpu"))
        neg_b = neg_b.to(torch.device("cpu"))
        test_pts.append((pos_a, pos_b, neg_a, neg_b))
    if args.test:
        val_stats=validation(args, model, test_pts, logger, 0, verbose=True)
        return val_stats
    print('test points are created')
    print('load only train data')

    #############train#################
    data_source = make_data_source(args)
    scheduler, opt = utils.build_optimizer(args, model.emb_model.parameters())
    train_stats = []
    print('args.n_epochs' +  str(args.n_epochs))
    for epoch in range(args.n_epochs):
        for batch_idx, (train_loss, train_acc) in enumerate(train(args, model, data_source, scheduler, opt, clf_opt)):
            print("epoch{}, batch_idx {}. Loss: {:.4f}. Training acc: {:.4f}".format(epoch, batch_idx, train_loss, train_acc), end="\n")
            train_stats.append((train_loss, train_acc))
            logger.add_scalar("Loss/train", train_loss, batch_idx)
            logger.add_scalar("Accuracy/train", train_acc, batch_idx)

        if epoch % args.eval_interval == 0:
            validation(args, model, test_pts, logger, epoch, verbose=True)

        torch.save(model.state_dict(), args.model_path)
        print("Model saved at: ", args.model_path)
def main(force_test=False):
    parser = (argparse.ArgumentParser(description='Order embedding arguments')
        if not HYPERPARAM_SEARCH else
        HyperOptArgumentParser(strategy='grid_search'))

    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()

    if force_test:
        args.test = True

    if HYPERPARAM_SEARCH:
        for i, hparam_trial in enumerate(args.trials(HYPERPARAM_SEARCH_N_TRIALS)):
            print("Running hyperparameter search trial", i)
            print(hparam_trial)
            train_loop(hparam_trial)
    else:
        train_loop(args)

if __name__ == '__main__':
    main()
