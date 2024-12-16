"""Train the order embedding model"""

# Set this flag to True to use hyperparameter optimization
# We use Testtube for hyperparameter tuning
HYPERPARAM_SEARCH = False
HYPERPARAM_SEARCH_N_TRIALS = None   # how many grid search trials to run
                                    #    (set to None for exhaustive search)
import argparse
import os
import torch
import torch.multiprocessing as mp

try:
     mp.set_start_method('spawn')
     mp.set_sharing_strategy('file_system')
    
except RuntimeError:
    pass

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from common import data
from common import models
from common import utils
if HYPERPARAM_SEARCH:
    from subgraph_matching.hyp_search import parse_encoder
else:
    from subgraph_matching.config import parse_encoder


def build_model(args):
    # build model
    if args.method_type == "order":
        model = models.OrderEmbedder(args.feature_size, args.hidden_dim, args)
    elif args.method_type == "mlp":
        model = models.BaselineMLP(args.feature_size, args.hidden_dim, args)
    model.to(utils.get_device())
    if os.path.exists(args.model_path):
        print('Model is loaded !!!!!!!!!')
        model.load_state_dict(torch.load(args.model_path,
            map_location=utils.get_device()))
    
    
    return model

def make_data_source(args):
    toks = args.dataset.split("-")
    if toks[0] == "syn":
        if len(toks) == 1 or toks[1] == "balanced":
            data_source = data.OTFSynDataSource(
                node_anchored=args.node_anchored)
        elif toks[1] == "imbalanced":
            data_source = data.OTFSynImbalancedDataSource(
                node_anchored=args.node_anchored)
        else:
            raise Exception("Error: unrecognized dataset")
    else:
        if 'darpha' in args.dataset or len(toks) == 1 or toks[1] == "balanced":
            data_source = data.DiskDataSource(args.dataset, args.data_identifier,
                node_anchored=args.node_anchored, feature_type=args.feature_type)
        elif toks[1] == "imbalanced":
            data_source = data.DiskImbalancedDataSource(toks[0],
                node_anchored=args.node_anchored)
        else:
            raise Exception("Error: unrecognized dataset")
    return data_source

