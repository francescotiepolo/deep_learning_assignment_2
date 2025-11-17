from lightning import LightningDataModule
from torch_geometric.loader import NeighborLoader
from torch_geometric.datasets import Planetoid

class DataModule(LightningDataModule):
    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        self.train_idx = dataset.train_mask.nonzero(as_tuple=True)[0]
        self.val_idx = dataset.val_mask.nonzero(as_tuple=True)[0]
        self.test_idx = dataset.test_mask.nonzero(as_tuple=True)[0]

        num_neighbors=[20]
        self.dataloader_kwargs = {
            'data': dataset[0],
            'num_neighbors': num_neighbors,
            'num_workers': 15,
        }

    def train_dataloader(self):
        return NeighborLoader(input_nodes=self.train_idx, batch_size=len(self.train_idx),
                              shuffle=True, **self.dataloader_kwargs)

    def val_dataloader(self):
        return NeighborLoader(input_nodes=self.val_idx, batch_size=len(self.val_idx), **self.dataloader_kwargs)

    def test_dataloader(self):
        return NeighborLoader(input_nodes=self.test_idx, batch_size=len(self.test_idx),
                              **self.dataloader_kwargs)

def get_dataset():
    dataset = Planetoid(root='data/Cora', name='Cora')
    return dataset
