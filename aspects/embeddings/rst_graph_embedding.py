from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch_geometric.nn import Node2Vec
from tqdm import tqdm

from aspects.embeddings.rst_trees_data import RSTTreesDataset
from aspects.utilities.settings import DEFAULT_OUTPUT_PATH

dataset = RSTTreesDataset(
    root=(DEFAULT_OUTPUT_PATH / 'reviews_Cell_Phones_and_Accessories-50000-docs' / 'our').as_posix(),
    spacy_model='en_core_web_lg'
)
data = dataset[0]
loader = DataLoader(torch.arange(data.num_nodes), batch_size=128, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Node2Vec(data.num_nodes, embedding_dim=128, walk_length=20, context_size=10, walks_per_node=10)
model, data = model.to(device), data.to(device)
optimizer = Adam(model.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2)


def train():
    model.train()
    total_loss = 0
    for subset in loader:
        optimizer.zero_grad()
        loss = model.loss(data.edge_index, subset.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


for epoch in tqdm(list(range(100)), desc='Epochs...'):
    loss = train()
    scheduler.step(loss)
    print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss))

model_path = Path(dataset.root) / Path(dataset.processed_file_names[0]).with_suffix(f'.{dataset.spacy_model}.model')
torch.save(model, model_path)
print(f'Model saved to: {model_path}')

dataset_path = Path(dataset.root) / Path(dataset.processed_file_names[0]).with_suffix(f'.{dataset.spacy_model}.dataset')
torch.save(dataset, dataset_path)
print(f'Dataset saved to: {dataset_path}')
