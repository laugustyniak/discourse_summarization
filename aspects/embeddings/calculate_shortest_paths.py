import argparse
import os

import networkx as nx
import yaml
from networkx.algorithms import shortest_paths as nx_sp
from tgem.tools import io
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Config path',
                        required=True)
    return parser.parse_args()


def calculate_shortest_paths_lengths(graph_path, out_path):
    graph = nx.read_gpickle(graph_path)
    if nx.is_directed(graph):
        graph = nx.DiGraph(graph)
    else:
        graph = nx.Graph(graph)

    test_sp = dict(nx_sp.shortest_path_length(graph))
    io.writers.export_obj_to_pickle(obj=test_sp, filepath=out_path)


def eval_snapshots(snapshots_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    for snap_id in tqdm(
            os.listdir(snapshots_path),
            desc='Snapshots',
            leave=False
    ):
        calculate_shortest_paths_lengths(
            graph_path=os.path.join(snapshots_path, snap_id),
            out_path=os.path.join(out_path, snap_id)
        )


def main():
    args = get_args()
    print('Loading configuration...')
    with open(args.config, 'r') as fobj:
        cfg = yaml.load(fobj, Loader=yaml.FullLoader)

    snapshots_path = cfg['directories']['snapshots']
    sp_out_path = cfg['directories']['sp']

    for ds_nm in tqdm(cfg['datasets'], desc='Datasets'):
        eval_snapshots(
            snapshots_path=os.path.join(snapshots_path, f'partial/{ds_nm}'),
            out_path=os.path.join(sp_out_path, f'partial/{ds_nm}')
        )
        eval_snapshots(
            snapshots_path=os.path.join(snapshots_path, f'full/{ds_nm}'),
            out_path=os.path.join(sp_out_path, f'full/{ds_nm}')
        )


if __name__ == '__main__':
    main()
