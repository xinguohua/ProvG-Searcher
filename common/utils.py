import math
import os
import pickle as pc
import random
from collections import Counter
from collections import defaultdict
import helper.create_candidate as cre_candi
import helper.create_pos_neg_dict as cd

import networkx as nx
import numpy as np
##### added by enes
import pandas as pd
import torch
import torch.optim as optim
from deepsnap.batch import Batch
from deepsnap.graph import Graph as DSGraph
from tqdm import tqdm

from helper import utilsDarpha

# from common import hash

base_dir = ''

import sys

sys.path.insert(0, 'helper/')
import dill
import helper.hash as g_hash

device_index = 0
nodes_type = None
nodes_data = None
nodes_data_test = None
nodes_data_train = None
memory_index = None
abstractType2array = None
procAdd = None
procName2Feature = None
procCandidatesTest = None
procCandidatesTrain = None
typeAbs = None
numberOfNeighK = None
global_args = None
numberOfFeature = -1

negQueryHashes1 = defaultdict(lambda: defaultdict(int))
posQueryHashes1 = defaultdict(lambda: defaultdict(int))

posQueryHashes = defaultdict(lambda: defaultdict(set))
hash2graph = dict()
posQueryHashStats = None
hash2seed = None


def load_datas(feature='ta1-theia-e3-official-6r', numberOfNeighK=None,
               args=None):
    global nodes_data, procCandidatesTest, procCandidatesTrain, numberOfNeighk, global_args, typeAbs
    global hash2graph, posQueryHashes, posQueryHashStats, hash2seed

    feature_dir = f'node_feature/{feature}/'
    abstarct_indexer = pc.load(open(f'{feature_dir}abstarct_indexer.pc', 'rb'))
    nodes_data = pd.read_csv(f'{feature_dir}nodes_data.csv')
    nodes_data = nodes_data.drop_duplicates(subset=['uuid'])
    nodes_data = nodes_data.set_index('uuid', drop = False)
    typeAbs = {}
    typeAbs['other'] = set(nodes_data.type)
    typeAbs['proc'] = set([tp for tp in typeAbs['other'] if '_Proc' in tp])
    typeAbs['file'] = set([tp for tp in typeAbs['other'] if '_File' in tp])
    typeAbs['other'] = typeAbs['other'].difference(typeAbs['proc'].union(typeAbs['file']))
    default_index = len(abstarct_indexer)
    nodes_data['type_index'] = nodes_data['type'].apply(
        lambda tp: abstarct_indexer.get(tp, default_index)
    )

    sampling_stats = f'data/{args.data_identifier}/test_neg_dict_{args.numberOfNeighK}.pc'
    if not os.path.exists(sampling_stats):
        print('first sampling stats will be created')
        print('this is a one time process for each dataset')
        cre_candi.run(args.numberOfNeighK, args.data_identifier, nodes_data)
        cd.run(nodes_data, args.data_identifier, args.numberOfNeighK)

    numberOfNeighk = numberOfNeighK
    global_args = args

    print('load data')


    data_dir = f'data/{feature}/'
    save_file = f'{data_dir}test_neg_dict_{numberOfNeighK}.pc'
    procCandidatesTest = pc.load(open(save_file, 'rb'))
    save_file = f'{data_dir}train_neg_dict_{numberOfNeighK}.pc'
    procCandidatesTrain = pc.load(open(save_file, 'rb'))
    print(save_file)

    hash2graph = pc.load(open(f'{data_dir}{numberOfNeighK}hash2graph.pkl', 'rb'))
    posQueryHashes = dill.load(open(f'{data_dir}{numberOfNeighK}posQueryHashes.pkl', 'rb'))
    posQueryHashStats = dill.load(open(f'{data_dir}{numberOfNeighK}posQueryHashStats.pkl', 'rb'))
    hash2seed = dill.load(open(f'{data_dir}{numberOfNeighK}hash2seed.pkl', 'rb'))
    print('data loaded')




def prepare_feature(feature='ta1-theia-e3-official-6r'):
    global nodes_data, abstractType2array, procAdd, procName2Feature
    feature_dir = f'node_feature/{feature}/'
    abstractType2array = pc.load(open(feature_dir + 'type2array.pc', 'rb'))
    procName2Feature = utilsDarpha.findProcFeature('gtfobins/*.md')
    procAdd = pc.load(open('gtfobins/procAdd.pkl', 'rb'))
    print('feature loaded')

def getNodeTypeNew(v):
    global abstractType2array
    row = nodes_data.loc[v]
    feat = abstractType2array[row.type].copy()
    return feat

def getProcUse(v):
    global nodes_data, procAdd, procName2Feature
    feat_proc = [0] * len(procAdd)
    try:
        row = nodes_data.loc[v]
        if '_Proc' in row.type:
            if type(row.path) == str:
                proc_name = row.path.split('/')[-1]
                if proc_name in procName2Feature:
                    for proc_feat in procName2Feature[proc_name]:
                        feat_proc[procAdd[proc_feat]] = 1
    except:
        pass

    return feat_proc

def create_neg_query(graphs, org_node, isTrain):
    global hash2graph, posQueryHashes, posQueryHashStats, hash2seed, negQueryHashes1

    global nodes_data_train, nodes_data_test, nodes_data
    global procCandidatesTest, procCandidatesTrain
    if isTrain:
        if nodes_data_train is None:
            nodes_data_train = nodes_data.loc[list(graphs.keys())]
        procCandidates_tmp = procCandidatesTrain
    else:
        if nodes_data_test is None:
            nodes_data_test = nodes_data.loc[list(graphs.keys())]
        procCandidates_tmp = procCandidatesTest
    dictKey = 'train' if isTrain else 'test'

    node_path = nodes_data.loc[org_node].path
    path_stat = posQueryHashStats[dictKey][node_path]
    # 其他节点同一的路径的子图
    candidate_paths = set(path_stat.keys()) - set(posQueryHashes[node_path][org_node])
    if len(candidate_paths) == 0:
        abstract = nodes_data.loc[org_node].type
        tmp_df = nodes_data[(nodes_data.type == abstract) & (nodes_data.path != node_path)]
        candidate_uuids = set(tmp_df.index.to_list()) & set(list(graphs.keys()))
        path_stat = dict()
        for candidate_path in set(tmp_df.loc[list(candidate_uuids)].path.to_list()):
            path_stat = {**path_stat, **posQueryHashStats[dictKey][candidate_path]}
        candidate_paths = set(path_stat.keys()) - set(posQueryHashes[node_path][org_node])

    # 核心
    if len(candidate_paths) > 0:
        weights = list(1 / path_stat[x] if path_stat[x] != 0 else 1e-8 for x in candidate_paths)
        neg_path_hash = random.choices(list(candidate_paths), weights=weights)[0]
        possible_seeds = list(hash2seed[neg_path_hash] & set(list(graphs.keys())))
        if len(possible_seeds) != 0:
            neg_seed = random.choices(possible_seeds)[0]
            initial_graph = hash2graph[neg_seed][neg_path_hash].copy()
            graph = graphs[neg_seed]
            min_size = min(4, len(graph))
            max_size = min(10, len(graph))
            size = random.randint(min_size, max_size)
            ne = findNeighPathFromPath(graph, initial_graph, size)
            if len(ne) > max(min_size, 1):
                return None, ne, 'new method', neg_seed
    # 兜底
    if org_node in procCandidates_tmp:
        cnd_dict = procCandidates_tmp[org_node]
        cnd, cnd_type = getCnd(cnd_dict)
        if cnd in graphs:
            if cnd_type == 'other':
                gr, ne = sample_graph_neigh_other(graphs[cnd], cnd)
                start_node = cnd

            if cnd_type == 'normal':
                cnds = [cnd]
                cnds.extend(list(cnd_dict['normal'][cnd]['cndNod']))
                gr, ne = sample_graph_neigh_same(graphs[cnd], cnds)
                start_node = cnds[0]

            if cnd_type == 'rand':
                g = graphs[org_node].copy()
                start_node = org_node
                possibleChanges = getPosChanges(g, org_node, procCandidates_tmp)
                if len(possibleChanges) > 0:
                    neigh = [org_node]
                    for possibleChange in possibleChanges:
                        seed_proc = list(possibleChange.keys())[0]
                        neigh.append(possibleChange[seed_proc])
                        g_tmp = graphs[seed_proc].copy()
                        g_tmp = nx.relabel_nodes(g_tmp, possibleChange)
                        g = nx.compose(g, g_tmp)
                    gr, ne = sample_graph_neigh_same(g, neigh)
                else:
                    gr, ne = random_change(g.copy(), org_node)
            return gr, ne, cnd_type, start_node
    return None, None, None, None


def random_change(g, org_node, maxChange=5):
    global typeAbs, nodes_data

    ins = set([edge[0] for edge in g.in_edges(org_node)])
    outs = set([edge[1] for edge in g.out_edges(org_node)])
    neighs = list(ins.union(outs))
    neighs = random.sample(neighs, min(maxChange, len(neighs)))
    sample_neigh = [org_node]
    ralable_dict = {}
    for neigh in neighs:
        neigh_type = nodes_data.loc[neigh].type
        for vl in typeAbs.values():
            if neigh_type in vl:
                break
        all_type = vl.copy()
        all_type.remove(neigh_type)
        new_uuid = nodes_data[nodes_data['type'].isin(all_type)].sample(1).index[0]
        ralable_dict[neigh] = new_uuid
        sample_neigh.append(new_uuid)

    g = nx.relabel_nodes(g, ralable_dict)

    return sample_graph_neigh_same(g, sample_neigh)


def find_pos_hashes(graph, node):
    for i in range(10000):
        graph, nodes = find_pos_hash_child(graph, node, recursiveCount=0)
        if graph is None:
            break


def find_pos_hash_child(graph, node, recursiveCount=0):
    c = 0

    if recursiveCount > 500:
        # print('COULD NOT FIND A UNIQUE POS QUERY FOR', node)
        return None, None

    graph, nodes = sample_graph_neigh(graph, node, findPath=True)
    if return_cond(graph, nodes, c):
        positiveHash = g_hash.get_hash_targetNodes(nodes, node, numberOfNeighk)
        proc_path = nodes_data.loc[node].path
        # posQueryHashes = defaultdict(lambda: defaultdict(dict))
        if not positiveHash in posQueryHashes[proc_path][node]:
            posQueryHashes[proc_path][node].add(positiveHash)
            # posQueryHashes[proc_path][positiveHash].add(node)
            hash2graph[node][positiveHash] = nodes
            #     lmt = np.mean(list(posQueryHashes[node].values())) + 2
            #     if posQueryHashes[node][positiveHash] > lmt:
        else:
            return find_pos_hash_child(graph, node, recursiveCount + 1)
            # posQueryHashes[node][positiveHash] += 1

        return graph, nodes


def find_pos_hashes_mp(data, graph, node):
    global nodes_data
    global numberOfNeighk

    nodes_data = data
    numberOfNeighk = 3
    g_hash.set_nodesdata(nodes_data)

    paths = []
    hashSet = set()
    for i in range(10000):
        positiveHash, nodes = find_pos_hash_child_mp(graph, node, hashSet, recursiveCount=0)
        if nodes is None:
            break
        paths.append((positiveHash, nodes))
        hashSet.add(positiveHash)
    return paths


def find_pos_hash_child_mp(graph, node, hashSet, recursiveCount=0):
    c = 0

    if recursiveCount > 500:
        # print('COULD NOT FIND A UNIQUE POS QUERY FOR', node)
        return None, None

    graph, nodes = sample_graph_neigh(graph, node, findPath=True)
    if return_cond(graph, nodes, c):
        positiveHash = g_hash.get_hash_targetNodes(nodes, node, numberOfNeighk)
        proc_path = nodes_data.loc[node].path
        # posQueryHashes = defaultdict(lambda: defaultdict(dict))
        if not positiveHash in hashSet:
            return positiveHash, nodes
        else:
            return find_pos_hash_child_mp(graph, node, hashSet, recursiveCount + 1)


def create_pos_query(graph, node, isTrain):
    global posQueryHashes1
    dictKey = 'train' if isTrain else 'test'
    node_path = nodes_data.loc[node].path
    path_stat = posQueryHashStats[dictKey][node_path]
    candidateHashs = set(posQueryHashes[node_path][node])
    if len(candidateHashs) > 0:
        weights = [1 / path_stat[x] if path_stat[x] != 0 else 0 for x in candidateHashs]
        if sum(weights) == 0:
            print('all zero weight:', candidateHashs)
        else:
            candidate_hash = random.choices(list(candidateHashs), weights=weights)[0]
            initial_graph = hash2graph[node][candidate_hash]
            min_size = min(4, len(graph))
            max_size = min(10, len(graph))
            size = random.randint(min_size, max_size)
            ne = findNeighPathFromPath(graph, initial_graph, size)
            return graph, ne
    return None, None


def getPosChanges(g, seed_proc, procCandidates_tmp):
    proc_can = set(procCandidates_tmp.keys())
    ins = set([edge[0] for edge in g.in_edges(seed_proc)])
    ins = ins.intersection(proc_can)
    outs = set([edge[1] for edge in g.out_edges(seed_proc)])
    outs = outs.intersection(proc_can)
    neigh = list(ins.union(outs))
    if len(neigh) == 0:
        return []

    random.shuffle(neigh)
    possibleChange = []
    for ngh in neigh:
        others = list(procCandidates_tmp[ngh]['other'].keys())
        if len(others) == 0:
            continue
        others = random.sample(others, 1)[0]
        possibleChange.append({others: ngh})
        if len(possibleChange) > 2:
            break

    return possibleChange


def getCnd(cnd_dict):
    cnd_proc = []  # we can use random weight, it will be the same
    cnd_proc += [cnd for cnd, vl in cnd_dict['other'].items() for _ in range(vl)]
    cnd_proc += [cnd for cnd, vl in cnd_dict['normal'].items() for _ in range(vl['val'])]
    cnd_proc += ['rand' for _ in range(math.ceil(cnd_dict['rand']))]
    if len(cnd_proc) == 0:
        return 'rand', 'rand'

    cnd = random.sample(cnd_proc, 1)[0]

    if cnd in cnd_dict['other']:
        return cnd, 'other'
    if cnd in cnd_dict['normal']:
        return cnd, 'normal'
    if cnd == 'rand':
        return cnd, 'rand'

    raise ValueError('Unexpected error in getCnd')


def check_one_degree(graph, neigh):
    sbr_grp = graph.subgraph(neigh)
    not_one_degre = [1 for node in sbr_grp.nodes if sbr_grp.out_degree(node) > 2]
    return len(not_one_degre) > 3


def return_cond(graph, neigh, c):
    return True
    one_degre = check_one_degree(graph, neigh)
    if c > 10 and one_degre:
        return True
    if c > 15:
        return True

    return False


def get_graph_nodes(graphs):
    start_node_id = random.choice(list(graphs.keys()))
    graph = graphs[start_node_id]
    return graph, start_node_id


def sample_graph_neigh(graph, start_node, findPath=False):
    neigh = [start_node]
    return graph, findNeignh(graph, neigh, findPath=findPath)


def sample_graph_neigh_other(graph, start_node):
    neigh = [start_node]
    return graph, findNeignh(graph, neigh)


def sample_graph_neigh_same(graph, neigh):
    return graph, findNeignh(graph, neigh)


def findNeignh(graph, neigh, findPath=False):
    min_size = min(4, len(graph))
    max_size = min(10, len(graph))

    start_node = neigh[0]
    size = random.randint(min_size, max_size)
    # Graph.subgraph neighbor
    initial_graph = graph.subgraph(neigh).copy()

    return findNeighPath(graph, initial_graph, start_node, size, findPath=findPath)


def findNeighPathFromPath(graph, initial_graph, size):
    frontier_edges = set()
    for node in initial_graph.nodes:
        frontier_edges = frontier_edges | set(graph.in_edges(node))
        frontier_edges = frontier_edges | set(graph.out_edges(node))

    frontier_edges = frontier_edges - set(initial_graph.edges)

    while len(list(initial_graph.nodes)) < size and frontier_edges:
        new_edge = random.choices(list(frontier_edges))[0]
        if new_edge in set(initial_graph.edges):
            continue
        frontier_edges = frontier_edges - set([new_edge])
        edge_feature = list(graph.get_edge_data(*new_edge).items())[0][1]
        initial_graph.add_edge(new_edge[0], new_edge[1], type_edge=edge_feature['type_edge'])
    return initial_graph


def findNeighPath(graph, initial_graph, start_node, size, findPath=False):
    # from initial graph we will continue adding edges
    # until reaching the predefined size
    # global numberOfNeighk
    degrees = graph.degree

    visited = set(list(initial_graph.nodes))
    frontiers = [start_node]
    for i in range(random.randint(1, numberOfNeighk)):
        if len(initial_graph) > size:
            return initial_graph
        tmp_front = set(graph.neighbors(frontiers[-1]))
        tmp_front = list(tmp_front - visited)
        if len(tmp_front) == 0:
            break
        new_node = random.choices(tmp_front, weights=(degrees[x] ** 2 for x in tmp_front))[0]
        frontiers.append(new_node)
        visited.add(new_node)
        edge = (frontiers[-2], frontiers[-1],)
        edge_feature = list(graph.get_edge_data(*edge).items())[0][1]
        initial_graph.add_edge(frontiers[-2], frontiers[-1], type_edge=edge_feature['type_edge'])

    g_reverse = graph.reverse()
    frontiers = [start_node]
    for i in range(random.randint(1, numberOfNeighk)):
        if len(initial_graph) > size:
            return initial_graph
        tmp_front = set(g_reverse.neighbors(frontiers[-1]))
        tmp_front = list(tmp_front - visited)
        if len(tmp_front) == 0:
            break
        new_node = random.choices(tmp_front, weights=(degrees[x] ** 2 for x in tmp_front))[0]
        frontiers.append(new_node)
        visited.add(new_node)
        edge = (frontiers[-1], frontiers[-2],)
        edge_feature = list(graph.get_edge_data(*edge).items())[0][1]
        initial_graph.add_edge(frontiers[-1], frontiers[-2], type_edge=edge_feature['type_edge'])

    if findPath:
        return initial_graph

    frontier_edges = set()
    for neig in initial_graph.nodes:
        frontier_edges = frontier_edges | set(graph.in_edges(neig))
        frontier_edges = frontier_edges | set(graph.out_edges(neig))

    frontier_edges = frontier_edges - set(initial_graph.edges)

    while len(list(initial_graph.nodes)) < size and frontier_edges:
        new_edge = random.choices(list(frontier_edges))[0]  # ,weights=(degrees[x[]]**3 for x in frontier_edges)
        if new_edge in set(initial_graph.edges):
            continue
        frontier_edges = frontier_edges - set([new_edge])
        edge_feature = list(graph.get_edge_data(*new_edge).items())[0][1]
        initial_graph.add_edge(new_edge[0], new_edge[1], type_edge=edge_feature['type_edge'])
    return initial_graph


def find_visited(graph, start_node, degrees):
    return set([start_node])


cached_masks = None


def vec_hash(v):
    global cached_masks
    if cached_masks is None:
        random.seed(2019)
        cached_masks = [random.getrandbits(32) for i in range(len(v))]
    # v = [hash(tuple(v)) ^ mask for mask in cached_masks]
    v = [hash(v[i]) ^ mask for i, mask in enumerate(cached_masks)]
    # v = [np.sum(v) for mask in cached_masks]
    return v


def wl_hash(g, dim=64, node_anchored=False):
    g = nx.convert_node_labels_to_integers(g)
    vecs = np.zeros((len(g), dim), dtype=np.int)
    if node_anchored:
        for v in g.nodes:
            if g.nodes[v]["anchor"] == 1:
                vecs[v] = 1
                break
    for i in range(len(g)):
        newvecs = np.zeros((len(g), dim), dtype=np.int)
        for n in g.nodes:
            newvecs[n] = vec_hash(np.sum(vecs[list(g.neighbors(n)) + [n]],
                                         axis=0))
        vecs = newvecs
    return tuple(np.sum(vecs, axis=0))


def gen_baseline_queries_rand_esu(queries, targets, node_anchored=False):
    sizes = Counter([len(g) for g in queries])
    max_size = max(sizes.keys())
    all_subgraphs = defaultdict(lambda: defaultdict(list))
    total_n_max_subgraphs, total_n_subgraphs = 0, 0
    for target in tqdm(targets):
        subgraphs = enumerate_subgraph(target, k=max_size,
                                       progress_bar=len(targets) < 10, node_anchored=node_anchored)
        for (size, k), v in subgraphs.items():
            all_subgraphs[size][k] += v
            if size == max_size: total_n_max_subgraphs += len(v)
            total_n_subgraphs += len(v)
    print(total_n_subgraphs, "subgraphs explored")
    print(total_n_max_subgraphs, "max-size subgraphs explored")
    out = []
    for size, count in sizes.items():
        counts = all_subgraphs[size]
        for _, neighs in list(sorted(counts.items(), key=lambda x: len(x[1]),
                                     reverse=True))[:count]:
            print(len(neighs))
            out.append(random.choice(neighs))
    return out


def enumerate_subgraph(G, k=3, progress_bar=False, node_anchored=False):
    ps = np.arange(1.0, 0.0, -1.0 / (k + 1)) ** 1.5
    # ps = [1.0]*(k+1)
    motif_counts = defaultdict(list)
    for node in tqdm(G.nodes) if progress_bar else G.nodes:
        sg = set()
        sg.add(node)
        v_ext = set()
        neighbors = [nbr for nbr in list(G[node].keys()) if nbr > node]
        n_frac = len(neighbors) * ps[1]
        n_samples = int(n_frac) + (1 if random.random() < n_frac - int(n_frac)
                                   else 0)
        neighbors = random.sample(neighbors, n_samples)
        for nbr in neighbors:
            v_ext.add(nbr)
        extend_subgraph(G, k, sg, v_ext, node, motif_counts, ps, node_anchored)
    return motif_counts


def extend_subgraph(G, k, sg, v_ext, node_id, motif_counts, ps, node_anchored):
    # Base case
    sg_G = G.subgraph(sg)
    if node_anchored:
        sg_G = sg_G.copy()
        nx.set_node_attributes(sg_G, 0, name="anchor")
        sg_G.nodes[node_id]["anchor"] = 1

    motif_counts[len(sg), wl_hash(sg_G,
                                  node_anchored=node_anchored)].append(sg_G)
    if len(sg) == k:
        return
    # Recursive step:
    old_v_ext = v_ext.copy()
    while len(v_ext) > 0:
        w = v_ext.pop()
        new_v_ext = v_ext.copy()
        neighbors = [nbr for nbr in list(G[w].keys()) if nbr > node_id and nbr
                     not in sg and nbr not in old_v_ext]
        n_frac = len(neighbors) * ps[len(sg) + 1]
        n_samples = int(n_frac) + (1 if random.random() < n_frac - int(n_frac)
                                   else 0)
        neighbors = random.sample(neighbors, n_samples)
        for nbr in neighbors:
            # if nbr > node_id and nbr not in sg and nbr not in old_v_ext:
            new_v_ext.add(nbr)
        sg.add(w)
        extend_subgraph(G, k, sg, new_v_ext, node_id, motif_counts, ps,
                        node_anchored)
        sg.remove(w)


def gen_baseline_queries_mfinder(queries, targets, n_samples=10000,
                                 node_anchored=False):
    sizes = Counter([len(g) for g in queries])
    # sizes = {}
    # for i in range(5, 17):
    #    sizes[i] = 10
    out = []
    for size, count in tqdm(sizes.items()):
        print(size)
        counts = defaultdict(list)
        for i in tqdm(range(n_samples)):
            graph, neigh = sample_neigh(targets, size)
            v = neigh[0]
            neigh = graph.subgraph(neigh).copy()
            nx.set_node_attributes(neigh, 0, name="anchor")
            neigh.nodes[v]["anchor"] = 1
            neigh.remove_edges_from(nx.selfloop_edges(neigh))
            counts[wl_hash(neigh, node_anchored=node_anchored)].append(neigh)

        for _, neighs in list(sorted(counts.items(), key=lambda x: len(x[1]),
                                     reverse=True))[:count]:
            print(len(neighs))
            out.append(random.choice(neighs))
    return out


device_cache = None


def get_device():
    global device_cache, device_index
    if device_cache is None:
        device_cache = torch.device("cuda:" + str(device_index)) if torch.cuda.is_available() \
            else torch.device("cpu")
    return device_cache


def parse_optimizer(parser):
    opt_parser = parser.add_argument_group()
    opt_parser.add_argument('--opt', dest='opt', type=str,
                            help='Type of optimizer')
    opt_parser.add_argument('--opt-scheduler', dest='opt_scheduler', type=str,
                            help='Type of optimizer scheduler. By default none')
    opt_parser.add_argument('--opt-restart', dest='opt_restart', type=int,
                            help='Number of epochs before restart (by default set to 0 which means no restart)')
    opt_parser.add_argument('--opt-decay-step', dest='opt_decay_step', type=int,
                            help='Number of epochs before decay')
    opt_parser.add_argument('--opt-decay-rate', dest='opt_decay_rate', type=float,
                            help='Learning rate decay ratio')
    opt_parser.add_argument('--lr', dest='lr', type=float,
                            help='Learning rate.')
    opt_parser.add_argument('--clip', dest='clip', type=float,
                            help='Gradient clipping.')
    opt_parser.add_argument('--weight_decay', type=float,
                            help='Optimizer weight decay.')


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95,
                              weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer


def v2w2v(v, feature_type):
    node_path = nodes_data.loc[v]['path']
    try:
        ftr = nodes_data_path2vecs.loc[node_path][feature_type]
    except:
        if str(node_path) == 'nan':
            ftr = np.zeros(30)
        else:
            raise
    return ftr


def getNodeType(v):
    global nodes_data, memory_index, numberOfFeature
    types = [0] * numberOfFeature
    try:
        types[nodes_data.loc[v]['type_index']] = 1
    except:
        pass
    return types

def feature_graphs_abstract(graphs, feature_type='type_basic', anchors=None):
    for anchor, g in zip(anchors, graphs):
        for v in g.nodes:
            node_feature = [float(v == anchor)]
            if feature_type.startswith('type_basic'):
                node_feature.extend(getNodeType(v))
            else:
                node_feature.extend(getNodeTypeNew(v))

            if 'Proc' in feature_type:
                node_feature.extend(getProcUse(v))

            node_feature = np.array(node_feature).astype(np.float32)
            g.nodes[v]["node_feature"] = torch.tensor(node_feature).float()

    return graphs


def feature_graphs_w2v(graphs, feature_type, anchors=None):
    for anchor, g in zip(anchors, graphs):
        for v in g.nodes:
            node_feature = [float(v == anchor)]
            node_feature.extend(v2w2v(v, feature_type))
            node_feature = np.array(node_feature).astype(np.float32)
            g.nodes[v]["node_feature"] = torch.tensor(node_feature).float()
    return graphs


def feature_graphs_combine(graphs, feature_type, anchors=None):
    for anchor, g in zip(anchors, graphs):
        for v in g.nodes:
            node_feature = [float(v == anchor)]
            node_feature.extend(getNodeType(v))
            node_feature.extend(v2w2v(v, feature_type))
            node_feature = np.array(node_feature).astype(np.float32)
            g.nodes[v]["node_feature"] = torch.tensor(node_feature).float()
    return graphs


def feature_graphs(graphs, feature_type, data_identifier, numberOfNeighK=None, anchors=None):
    global nodes_data
    if type(nodes_data) == type(None):
        print('load data inside batch!!!')
        load_datas(feature=data_identifier, numberOfNeighK=numberOfNeighK)

    splitted = feature_type.split('_')
    if feature_type.startswith('type'):
        graphs = feature_graphs_abstract(graphs, feature_type=feature_type, anchors=anchors)

    elif len(splitted) == 2:
        graphs = feature_graphs_w2v(graphs, feature_type, anchors=anchors)

    else:
        feature_type = splitted[0] + '_' + splitted[1]
        graphs = feature_graphs_combine(graphs, feature_type, anchors=anchors)

    return graphs


def batch_nx_graphs(graphs):
    batch = Batch.from_data_list([DSGraph(g.to_undirected()) for g in graphs])
    batch = batch.to(get_device())
    return batch

