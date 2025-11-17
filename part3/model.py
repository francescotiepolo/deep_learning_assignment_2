import torch
from torch import nn
import torch.nn.functional as F

from graph_cnn import MatrixGraphConvolution, MessageGraphConvolution, GraphAttention

class GraphNN(nn.Module):
    def __init__(self, config, dataset):
        super(GraphNN, self).__init__()
        self.config = config
        self.layers = []

        self.activation_fn = F.sigmoid
        graph_layer = {
            'gcn': MessageGraphConvolution,
            'matrix-gcn': MatrixGraphConvolution,
            'gat': GraphAttention,
        }[self.config.model]

        self.layers = nn.ModuleList()
        self.layers.append(graph_layer(dataset.num_features, config.hidden_dim))
        for _ in range(config.n_layers - 2):
            self.layers.append(nn.Dropout(self.config.dropout))
            self.layers.append(graph_layer(config.hidden_dim, config.hidden_dim))

        self.layers.append(graph_layer(config.hidden_dim, dataset.num_classes))
        self.dropout = config.dropout

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        return optimizer

    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = self.activation_fn(x)
        x = self.layers[-1](x, edge_index)
        return x