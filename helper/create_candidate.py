import pickle as pc
import pandas as pd
import math
from collections import defaultdict
import random


def run(numberOfLAyerK, dataIndentifier, nodes_data, min_length = 5, max_length = 2000):
    for testTrain in ['test', 'train']:
        dataset_train_path = f'data/{dataIndentifier}/k_{numberOfLAyerK}{testTrain}.pt'
        with open(dataset_train_path, 'rb') as f:
            dataset_train = pc.load(f)
        dataset_tmp = {k: v for k, v in dataset_train.items() if min_length < len(v) < max_length}
        print(f"Original dataset size: {len(dataset_train)}, Filtered dataset size: {len(dataset_tmp)}")
        del dataset_train

        # Filter nodes data based on dataset_tmp keys
        df = nodes_data[nodes_data['uuid'].isin(list(dataset_tmp.keys()))]
        proc_uuids = set(df['uuid'].values)
        proc_diffs = calculate_proc_diffs(nodes_data, dataset_tmp, proc_uuids)
        same_path_candidates = extract_same_path_candidates(proc_diffs)
        same_abs_candidates = extract_same_abs_candidates(dataset_tmp, nodes_data, proc_uuids)
        proc_candidates, revDict = process_candidates(proc_uuids, same_path_candidates, same_abs_candidates)
        optimize_candidate_distribution(proc_uuids, revDict, proc_candidates)
        compute_candidate_statistics(proc_candidates)

        save_file = f'data/{dataIndentifier}/{testTrain}_neg_dict_{numberOfLAyerK}.pc'
        pc.dump(proc_candidates, open(save_file, 'wb'))


def get_neigh(g, seed_proc, tip, proc_uuids, nodes_data):
    ins = set([edge[0] for edge in g.in_edges(seed_proc)])
    ins = ins.intersection(proc_uuids)
    outs = set([edge[1] for edge in g.out_edges(seed_proc)])
    outs = outs.intersection(proc_uuids)
    df = nodes_data[nodes_data['uuid'].isin(list(ins.union(outs)))]
    return df, set(df[tip].values)


def get_neigh_with_path(df, paths):
    return set(df[df['path'].isin(paths)]['uuid'].values)





def calculate_proc_diffs(nodes_data, dataset_tmp, proc_uuids):
    """
    Calculate neighbor differences for nodes grouped by unique paths.

    Parameters:
        nodes_data (pd.DataFrame): A DataFrame containing node features with columns 'uuid', 'type', and 'path'.
        dataset_tmp (dict): A dictionary where keys are node UUIDs and values are node-related data.
        proc_uuids (set): A set of all process UUIDs used to constrain neighbor computations.

    Returns:
        dict: A dictionary where keys are paths and values are nested dictionaries capturing node differences.
    """

    def get_neighbors_for_path(proc_ids):
        """
        Get neighbor information for all nodes in a given path.
        """
        return {
            node_id: get_neigh(
                dataset_tmp[node_id],
                node_id,
                'path',
                proc_uuids,
                nodes_data
            )
            for node_id in proc_ids
        }

    def compute_node_differences(proc_ids, neighs):
        """
        Compute differences in neighbor sets for all node pairs.
        """
        node_diffs = {}
        for i, node_id in enumerate(proc_ids):
            node_diffs[node_id] = {}
            current_neighbors = neighs[node_id][1]
            for j, compare_id in enumerate(proc_ids):
                if i == j:
                    continue
                compare_neighbors = neighs[compare_id][1]
                diffs = compare_neighbors.difference(current_neighbors)
                if diffs:
                    node_diffs[node_id][compare_id] = get_neigh_with_path(
                        neighs[compare_id][0], diffs
                    )
        return node_diffs

    df = nodes_data[nodes_data['uuid'].isin(dataset_tmp.keys())]
    proc_type = [tp for tp in set(nodes_data.type) if '_Proc' in tp]
    prc_df = df[df['type'].isin(proc_type)]
    uniq_paths = prc_df['path'].dropna().unique()
    proc_diffs = {}

    for proc_path in uniq_paths:
        # Get node IDs for the current path
        same_proc_ids = prc_df[prc_df['path'] == proc_path]['uuid'].values

        # Get neighbors for nodes in the current path
        neighs = get_neighbors_for_path(same_proc_ids)

        # Compute differences between node pairs
        proc_diffs[proc_path] = compute_node_differences(same_proc_ids, neighs)

    return proc_diffs


def extract_same_path_candidates(proc_diffs):
    # Example data structure for `proc_diffs`:
    # {
    #     'path1': {
    #         'procA': {
    #             'procB': {'node1', 'node2'},
    #             'procC': {'node3'}
    #         },
    #         'procD': {
    #             'procE': {'node4'}
    #         }
    #     },
    #     'path2': {
    #         'procX': {
    #             'procY': {'node5', 'node6'}
    #         }
    #     }
    # }

    # Example data structure for `samePathCandidates`:
    # {
    #     'procA': {
    #         'procB': {'node1', 'node2'},
    #         'procC': {'node3'}
    #     },
    #     'procD': {
    #         'procE': {'node4'}
    #     },
    #     'procX': {
    #         'procY': {'node5', 'node6'}
    #     }
    # }
    samePathCandidates = {}

    # Populate samePathCandidates by processing proc_diffs
    for path, procs in proc_diffs.items():
        for proc, candidateProcs in procs.items():
            for candidateProc, candidateNodes in candidateProcs.items():
                if proc not in samePathCandidates:
                    samePathCandidates[proc] = {}
                samePathCandidates[proc][candidateProc] = candidateNodes
    return samePathCandidates


def extract_same_abs_candidates(dataset_tmp, nodes_data, proc_uuids):
    """
    Extract same-abstract candidates for each process path based on type and path.

    Parameters:
        dataset_tmp (dict): A dictionary where keys are process paths and values are related data.
        nodes_data (pd.DataFrame): A DataFrame containing node attributes with at least 'uuid', 'type', and 'path' columns.
        proc_uuids (set): A set of valid process UUIDs for filtering.

    Returns:
        dict: A dictionary mapping each process path to its same-abstract candidates.
    """
    # Filter nodes_data to include only rows with UUIDs in proc_uuids
    nodes_data_proc = nodes_data[nodes_data['uuid'].isin(proc_uuids)]

    # Initialize dictionaries
    same_abs_candidates = {}
    pathCand = {}

    # Process each path in dataset_tmp
    for procId in dataset_tmp:
        # Retrieve process information
        proc_info = nodes_data_proc[nodes_data_proc['uuid'] == procId].iloc[0]
        path = proc_info.path

        # Check if path is already cached
        if path in pathCand:
            abs_procs = pathCand[path]
        else:
            # Find candidates of the same type but with different paths
            abs_procs = nodes_data_proc[nodes_data_proc['type'] == proc_info.type]
            abs_procs = abs_procs[abs_procs['path'] != proc_info.path].uuid.values
            pathCand[path] = abs_procs

        # Add candidates if they exist
        if len(abs_procs) > 0:
            same_abs_candidates[procId] = abs_procs

    return same_abs_candidates


def process_candidates(proc_uuids, samePathCandidates, sameAbsCandidates):
    # combine and create count
    procCandidates = {}
    revDict = defaultdict(dict)
    NFQ = 40
    for proc_uuid in proc_uuids:
        procCandidates[proc_uuid] = {'other': {}, 'normal': {}}

        if proc_uuid in sameAbsCandidates and proc_uuid in samePathCandidates:
            baseValues = NFQ / 2
        else:
            baseValues = 2 * NFQ / 3

        if proc_uuid in sameAbsCandidates:
            vl = math.ceil(baseValues / len(sameAbsCandidates[proc_uuid]))
            possibleProcs = []
            for cnd in sameAbsCandidates[proc_uuid]:
                possibleProcs.extend([cnd] * vl)
            possibleProcs = random.sample(possibleProcs, int(baseValues))

            for cnd in set(possibleProcs):
                vl = possibleProcs.count(cnd)
                procCandidates[proc_uuid]['other'][cnd] = vl
                revDict[cnd][proc_uuid] = vl

        if proc_uuid in samePathCandidates:
            vl = math.ceil(baseValues / len(samePathCandidates[proc_uuid]))
            possibleProcs = []
            for cnd in samePathCandidates[proc_uuid]:
                possibleProcs.extend([cnd] * vl)
            possibleProcs = random.sample(possibleProcs, int(baseValues))

            for cnd in set(possibleProcs):
                vl = possibleProcs.count(cnd)
                procCandidates[proc_uuid]['normal'][cnd] = {'val': vl, 'cndNod': samePathCandidates[proc_uuid][cnd]}
                revDict[cnd][proc_uuid] = vl
    return procCandidates, revDict


def optimize_candidate_distribution(candiIds, revDict, procCandidates, NFQ=40):
    for candi in candiIds:
        getMultiples = []
        for pr, vl in revDict[candi].items():
            getMultiples.extend([pr] * vl)
        if len(getMultiples) < 2 * NFQ:
            continue

        getMultiplesNew = random.sample(getMultiples, 2 * NFQ)
        for pr in set(getMultiples):
            cnt = getMultiplesNew.count(pr)
            if cnt == 0:
                revDict[candi].pop(pr, None)
                if candi in procCandidates[pr]['other']:
                    procCandidates[pr]['other'].pop(candi, None)
                else:
                    procCandidates[pr]['normal'].pop(candi, None)

            else:
                revDict[candi][pr] = cnt
                if candi in procCandidates[pr]['other']:
                    procCandidates[pr]['other'][candi] = cnt
                else:
                    procCandidates[pr]['normal'][candi]['val'] = getMultiples.count(pr)


def compute_candidate_statistics(procCandidates, NFQ=40):
    rand_lmt = []
    for proc_uuid in procCandidates:
        other_tot = sum(procCandidates[proc_uuid]['other'].values())
        same_tot = sum([vl['val'] for vl in procCandidates[proc_uuid]['normal'].values()])
        rnd_lmt = max(0, (NFQ / 5) * 4 - other_tot - same_tot)
        rand_lmt.append(rnd_lmt)
        procCandidates[proc_uuid]['rand'] = rnd_lmt
