import os
from collections import defaultdict
from datetime import datetime

import torch
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, Batch

from common import utils
from common.models import build_model

USE_ORCA_FEATS = False  # whether to use orca motif counts along with embeddings
MAX_MARGIN_SCORE = 1e9  # a very large margin score to given orca constraints

from sklearn.metrics import confusion_matrix
import numpy as np


def validation(args, model, test_pts, logger, epoch, verbose=False):
    # test on new motifs
    model.eval()
    all_emb_as, all_emb_bs = [], []
    all_raw_preds, all_preds, all_labels = [], [], []
    for pos_a, pos_b, neg_a, neg_b in test_pts:
        if pos_a:
            pos_a = pos_a.to(utils.get_device())
            pos_b = pos_b.to(utils.get_device())
        neg_a = neg_a.to(utils.get_device())
        neg_b = neg_b.to(utils.get_device())
        labels = torch.tensor([1] * (pos_a.num_graphs if pos_a else 0) +
                              [0] * neg_a.num_graphs).to(utils.get_device())
        with torch.no_grad():
            emb_neg_a, emb_neg_b = (model.emb_model(neg_a),
                                    model.emb_model(neg_b))
            if pos_a:
                emb_pos_a, emb_pos_b = (model.emb_model(pos_a),
                                        model.emb_model(pos_b))
                emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
                emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)
            else:
                emb_as, emb_bs = emb_neg_a, emb_neg_b

            all_emb_as.extend(emb_as)
            all_emb_bs.extend(emb_bs)
            pred = model(emb_as, emb_bs)  # embeds
            raw_pred = model.predict(pred)  # diff between embeds
            if USE_ORCA_FEATS:
                import orca
                import matplotlib.pyplot as plt
                def make_feats(g):
                    counts5 = np.array(orca.orbit_counts("node", 5, g))
                    for v, n in zip(counts5, g.nodes):
                        if g.nodes[n]["node_feature"][0] > 0:
                            anchor_v = v
                            break
                    v5 = np.sum(counts5, axis=0)
                    return v5, anchor_v

                for i, (ga, gb) in enumerate(zip(neg_a.G, neg_b.G)):
                    (va, na), (vb, nb) = make_feats(ga), make_feats(gb)
                    if (va < vb).any() or (na < nb).any():
                        raw_pred[pos_a.num_graphs + i] = MAX_MARGIN_SCORE

            if args.method_type == "order":
                pred = model.clf_model(torch.sigmoid(raw_pred.unsqueeze(1)))
                pred = pred.argmax(dim=-1)  # model from diff between embeds
                raw_pred *= -1  # close 0 for pozitive, min for negative
            elif args.method_type == "ensemble":
                pred = torch.stack([m.clf_model(
                    raw_pred.unsqueeze(1)).argmax(dim=-1) for m in model.models])
                for i in range(pred.shape[1]):
                    print(pred[:, i])
                pred = torch.min(pred, dim=0)[0]
                raw_pred *= -1
            elif args.method_type == "mlp":
                raw_pred = raw_pred[:, 1]
                pred = pred.argmax(dim=-1)
        all_raw_preds.append(raw_pred)
        all_preds.append(pred)
        all_labels.append(labels)
    pred = torch.cat(all_preds, dim=-1)
    labels = torch.cat(all_labels, dim=-1)
    raw_pred = torch.cat(all_raw_preds, dim=-1)
    acc = torch.mean((pred == labels).type(torch.float))  # getting from model
    prec = (torch.sum(pred * labels).item() / torch.sum(pred).item() if
            torch.sum(pred) > 0 else float("NaN"))
    recall = (torch.sum(pred * labels).item() /
              torch.sum(labels).item() if torch.sum(labels) > 0 else
              float("NaN"))
    labels = labels.detach().cpu().numpy()
    raw_pred = raw_pred.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()

    auroc = roc_auc_score(labels, raw_pred)  # getting from e

    avg_prec = average_precision_score(labels, raw_pred)
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
    if verbose:
        import matplotlib.pyplot as plt
        lr_fpr, lr_tpr, threshold = roc_curve(labels, raw_pred)
        plt.plot(lr_fpr, lr_tpr)
        plt.xlabel("Recall")
        plt.ylabel("Precision")

    print("\n{}".format(str(datetime.now())))
    print("Validation. Epoch {}. Acc: {:.4f}. "
          "P: {:.4f}. R: {:.4f}. AUROC: {:.4f}. AP: {:.4f}.\n     "
          "TN: {}. FP: {}. FN: {}. TP: {}".format(epoch,
                                                  acc, prec, recall, auroc, avg_prec,
                                                  tn, fp, fn, tp))

    if not args.test:
        logger.add_scalar("Accuracy/test", acc, epoch)
        logger.add_scalar("Precision/test", prec, epoch)
        logger.add_scalar("Recall/test", recall, epoch)
        logger.add_scalar("AUROC/test", auroc, epoch)
        logger.add_scalar("AvgPrec/test", avg_prec, epoch)
        logger.add_scalar("TP/test", tp, epoch)
        logger.add_scalar("TN/test", tn, epoch)
        logger.add_scalar("FP/test", fp, epoch)
        logger.add_scalar("FN/test", fn, epoch)
        print("Saving {}".format(args.model_path))
        torch.save(model.state_dict(), args.model_path)
    return labels, raw_pred



def test(args, model, test_pts, logger, epoch, verbose=False):
    model.eval()
    all_emb_as, all_emb_bs = [], []
    all_raw_preds, all_preds, all_labels = [], [], []
    for pos_a, pos_b in test_pts:
        if pos_a:
            pos_a = pos_a.to(utils.get_device())
            pos_b = pos_b.to(utils.get_device())
        with torch.no_grad():
            if pos_a:
                emb_pos_a, emb_pos_b = (model.emb_model(pos_a),
                                        model.emb_model(pos_b))
                all_emb_as.extend(emb_pos_a)
                all_emb_bs.extend(emb_pos_b)
                pred = model(emb_pos_a, emb_pos_b)  # embeds
                raw_pred = model.predict(pred)  # diff between embeds

            if args.method_type == "order":
                pred = model.clf_model(torch.sigmoid(raw_pred.unsqueeze(1)))
                pred = pred.argmax(dim=-1)
            elif args.method_type == "ensemble":
                pred = torch.stack([m.clf_model(
                    raw_pred.unsqueeze(1)).argmax(dim=-1) for m in model.models])
                for i in range(pred.shape[1]):
                    print(pred[:, i])
                pred = torch.min(pred, dim=0)[0]
            elif args.method_type == "mlp":
                raw_pred = raw_pred[:, 1]
                pred = pred.argmax(dim=-1)
        all_raw_preds.append(raw_pred)
        all_preds.append(pred)
    pred = torch.cat(all_preds, dim=-1)
    return pred


# 设置参数
class Args:
    def __init__(self):
        self.method_type = "order"  # 可选: "order", "ensemble", "mlp"
        self.test = False
        self.model_path = "../ckpt/ta1-theia-e3-official-6r_model.pt"
        self.feature_size = 54
        self.hidden_dim = 256
        self.dropout = 0.0
        self.n_layers = 4
        self.conv_type = 'SAGE_typed'
        self.skip = 'learnable'
        self.n_edge_type = 5
        self.pool = 'mean'
        self.margin = 0.1


def create_single_graph(num_nodes, num_edges):
    """
    构造一个单个 PyTorch Geometric 格式的图数据。
    """
    # 随机生成节点特征 (54维特征)
    node_features = torch.randn((num_nodes, 54))

    # 随机生成边索引 (2 x num_edges)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    # 构建 PyTorch Geometric Data 对象
    data = Data(x=node_features, edge_index=edge_index)
    data.node_feature = node_features

    type_edge = torch.randint(0, 5, (num_edges,))
    data.type_edge = type_edge

    return data


def test_model(args, test_pts):
    """
    加载训练好的模型，并在测试数据上运行验证。
    参数:
        args: 包含配置信息的对象 (如 model_path, method_type, dataset 等)
        test_pts: 测试数据点 (pos_a, pos_b, neg_a, neg_b)
    返回:
        验证结果，包括标签和预测值
    """
    # 确保日志目录存在
    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))
    if not os.path.exists("plots/"):
        os.makedirs("plots/")

    # 初始化日志记录器
    record_keys = ["conv_type", "n_layers", "hidden_dim",
                   "margin", "dataset", "max_graph_size", "skip"]
    args_str = ".".join(["{}={}".format(k, v)
                         for k, v in sorted(vars(args).items()) if k in record_keys])
    logger = SummaryWriter(comment=args_str)

    # 加载设备
    device = utils.get_device()

    # 构建模型
    print("Building model...")
    model = build_model(args)
    model = model.to(device)

    # 加载保存的权重
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path {args.model_path} does not exist.")
    print(f"Loading trained model from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # 执行验证
    print("Running validation on test dataset...")
    epoch = 0  # 测试阶段，epoch 设置为 0
    raw_pred = test(args, model, test_pts, logger, epoch, verbose=True)

    # 输出结果
    print("Validation Complete.")
    print("Raw Predictions: ", raw_pred)
    return raw_pred


def generate_test_pts(batch_size=4):
    """
    生成测试数据，符合 (pos_a, pos_b, neg_a, neg_b) 格式。
    """
    test_pts = []

    for _ in range(batch_size):
        # 生成正例图
        pos_a = Batch.from_data_list([create_single_graph(20, 50) for _ in range(1)])
        pos_b = Batch.from_data_list([create_single_graph(20, 50) for _ in range(1)])

        test_pts.append((pos_a, pos_b))
    return test_pts


# 主函数入口
if __name__ == "__main__":
    args = Args()

    print("Generating test data...")
    test_pts = generate_test_pts()

    print("Starting model testing...")
    raw_pred = test_model(args, test_pts)
    print("Final Raw Predictions: ", raw_pred)
