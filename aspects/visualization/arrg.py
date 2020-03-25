import sys
from pathlib import Path

import networkx as nx
import streamlit as st

sys.path.append('/datasets/sentiment/aspects/sentiment-backend/')

from aspects.data_io import serializer
from aspects.utilities.data_paths import ExperimentPaths

RESULTS_PATH = Path('/datasets/sentiment/aspects/sentiment-backend/results')

# TODO: to uncomment must no display variable in linux must be fixed, not needed for now

# it doesn't work
# from os import system
# system('export DISPLAY=:0.0')

# st.title('Draw Discourse Tree')
# input_text_for_discourse_tree = st.text_area(
#     "Text to create example Discourse Tree",
#     "Top card doesn't work / Bottom card works. Bought the top card June 18 2018 for a Nintendo Switch."
# )
# png_file = jupyter_draw_nltk_tree(
#     tree=extract_discourse_tree(input_text_for_discourse_tree),
#     f_name='discourse_tree_test'
# )
# discourse_tree_image = Image.open(png_file)
# st.image(discourse_tree_image)

st.title('ARRG Visualisations')

results_dirs = {
    p.name: p
    for p
    in RESULTS_PATH.glob('*')
}
st.sidebar.title('Which dataset do you want to analyze?')
results_dir_name = st.sidebar.selectbox('', sorted(results_dirs.keys()), index=len(results_dirs) - 1)

st.header('Available files and directories for:')
st.info(results_dir_name)
st.write([p.name for p in results_dirs[results_dir_name].glob('*')])
paths = ExperimentPaths('', RESULTS_PATH / results_dir_name)

st.header('Discourse Trees Data Frame structure')
# discourse_tree_df_cache = st.cache(serializer.load)
# discourse_tree_df = discourse_tree_df_cache(paths.discourse_trees_df)
# st.write(discourse_tree_df.sample(5))

aspect_sentiments = dict(serializer.load(paths.aspect_sentiments))
st.header('Aspect sentiments')
# st.write(aspect_sentiments)

# arrg = serializer.load(paths.aspect_to_aspect_graph)
# arrg_dot = nx.nx_pydot.to_pydot(arrg)
# st.graphviz_chart(arrg_dot.to_string())

aht = serializer.load(paths.aspect_hierarchical_tree)
aht_dot = nx.nx_pydot.to_pydot(aht)
st.graphviz_chart(aht_dot.to_string())
