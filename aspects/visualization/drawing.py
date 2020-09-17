from pathlib import Path
from typing import Tuple, Union, Optional

import matplotlib.pyplot as plt
import mlflow
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


def draw_tree(tree: nx.Graph, fig_path: Optional[Union[str, Path]], figsize: Tuple[int, int] = (50, 30)):
    pos = graphviz_layout(tree, prog='dot')
    plt.figure(figsize=figsize)
    # TODO: can be parametrized later, work for graph 50-100 nodes
    nx.draw(tree, pos, with_labels=False, arrows=True, node_size=2500, font_size=36)
    text = nx.draw_networkx_labels(tree, pos, with_labels=False, node_size=2500, font_size=36)
    for _, t in text.items():
        t.set_rotation(45)
    if fig_path:
        graph_viz_png = Path(fig_path).with_suffix('.png').as_posix()
        plt.savefig(graph_viz_png)
        mlflow.log_artifact(graph_viz_png)
