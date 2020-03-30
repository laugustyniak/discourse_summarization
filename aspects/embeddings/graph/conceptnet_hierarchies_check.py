from aspects.data.conceptnet.utils import load_english_graph
from aspects.data_io import serializer
from aspects.graph.convert import networkx_2_graph_tool
from aspects.utilities import settings
from aspects.utilities.data_paths import ExperimentPaths

if __name__ == '__main__':
    conceptnet_graph = load_english_graph()

    experiment_paths = ExperimentPaths(
        input_path='',
        output_path=settings.DEFAULT_OUTPUT_PATH / 'reviews_Cell_Phones_and_Accessories-50000-docs',
        experiment_name='our'
    )
    aspect_graph = serializer.load(experiment_paths.aspect_to_aspect_graph)
    aspect_graph = networkx_2_graph_tool(aspect_graph)

    print(aspect_graph)
