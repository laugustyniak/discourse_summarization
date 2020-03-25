import pickle
from pathlib import Path

import spacy
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx
from tqdm import tqdm


class RSTTreesDataset(InMemoryDataset):
    def __init__(self, root, spacy_model='en_core_web_lg'):
        self.spacy_model = spacy_model
        self.nlp = spacy.load(self.spacy_model, disable=['tagger', 'parser', 'ner'])
        super(RSTTreesDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['aspect_2_aspect_graph.pkl']

    @property
    def processed_file_names(self):
        return [f'aspect_2_aspect_graph-{self.spacy_model}.pt']

    def download(self):
        pass
        # Download to `self.raw_dir`.

    def process(self):
        # Read data into huge `Data` list.
        with open(Path(self.root) / self.raw_file_names[0], 'rb') as f:
            aspect_to_aspect_graph = pickle.load(f)
        data = from_networkx(aspect_to_aspect_graph)
        nodes = aspect_to_aspect_graph.nodes()
        nodes_mapping = dict(zip(nodes, range(0, aspect_to_aspect_graph.number_of_nodes())))

        aspect_embeddings = [self.nlp(aspect).vector for aspect in tqdm(nodes, desc='Generating aspects embeddings...')]
        node_features = torch.tensor(aspect_embeddings, dtype=torch.float)

        data.nodes_mapping = nodes_mapping
        data.x = node_features
        data_list = [data]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
