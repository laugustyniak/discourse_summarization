from collections import namedtuple
from itertools import product

import pandas as pd
from plotly.offline import iplot
from sklearn.manifold import TSNE

SymbolColor = namedtuple('ColorSymbol', 'symbol, color')

COLOR_PALETTE = [
    'rgb(253,174,97)',
    'rgb(215,48,39)',
    'rgb(166,217,106)',
    'rgb(26,152,80)',
    'rgb(0,139,139)',
    'rgb(0,191,255)',
    'rgb(0,0,128)',
    'rgb(138,43,226)',
    'rgb(0,0,0)',
]

SYMBOL_COLOR = [
    SymbolColor(cs[0], cs[1])
    for cs
    in product(range(33), COLOR_PALETTE)
]


def get_tsne(df: pd.DataFrame, intent_col: str = 'aspect', tooltip_col: str = 'aspect', tsne=None) -> pd.DataFrame:
    if tsne is None:
        tsne = TSNE(n_components=2, init='random', random_state=0, perplexity=30)
    tsne_coords = tsne.fit_transform(df.embedding.tolist())
    return pd.DataFrame(dict(
        x=tsne_coords[:, 0],
        y=tsne_coords[:, 1],
        intent=df[intent_col],
        tooltip=df[tooltip_col],
        cluster=df.cluster
    ))


def draw(df: pd.DataFrame):
    layout = {
        'autosize': False,
        'width': 1500,
        'height': 1500,
        'margin': {
            'l': 50,
            'r': 50,
            'b': 100,
            't': 100,
            'pad': 4
        }
    }

    data = [
        {
            'x': sub_df.x,
            'y': sub_df.y,
            'text': sub_df.tooltip,
            'marker': {
                'symbol': SYMBOL_COLOR[cluster % len(SYMBOL_COLOR)].symbol,
                'color': SYMBOL_COLOR[cluster % len(SYMBOL_COLOR)].color,
                'size': 15
            },
            'mode': 'markers',
            'name': cluster
        }
        for cluster, sub_df
        in df.groupby(by='cluster')
    ]

    iplot({
        'data': data,
        'layout': layout
    })
