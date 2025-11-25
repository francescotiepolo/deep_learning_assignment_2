import torch.nn as nn
import torch

import torch.nn.functional as F
from torch_geometric.utils import add_self_loops


class MatrixGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(MatrixGraphConvolution, self).__init__()
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.B = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.B)

    def make_adjacency_matrix(self, edge_index, num_nodes):
        """
        Creates adjacency matrix from edge index.

        :param edge_index: [source, destination] pairs defining directed edges nodes. dims: [2, num_edges]
        :param num_nodes: number of nodes in the graph.
        :return: adjacency matrix with shape [num_nodes, num_nodes]

        Hint: A[i,j] -> there is an edge from node j to node i
        """
        sources, destinations = edge_index
        adjacency_matrix = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
        adjacency_matrix[destinations, sources] = 1.0
        return adjacency_matrix

    def make_inverted_degree_matrix(self, edge_index, num_nodes):
        """
        Creates inverted degree matrix from edge index.

        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :param num_nodes: number of nodes in the graph.
        :return: inverted degree matrix with shape [num_nodes, num_nodes]. Set degree of nodes without an edge to 1.
        """
        degree_vector = self.make_adjacency_matrix(edge_index, num_nodes).sum(dim=1)
        inverted_degree_vector = 1.0 / torch.clamp(degree_vector, min=1.0)
        inverted_degree_matrix = torch.diag(inverted_degree_vector)
        return inverted_degree_matrix

    def forward(self, x, edge_index):
        """
        Forward propagation for GCNs using efficient matrix multiplication.

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :return: activations for the GCN
        """
        A = self.make_adjacency_matrix(edge_index, x.size(0))
        D_inv = self.make_inverted_degree_matrix(edge_index, x.size(0))
        out = F.linear(D_inv @ A @ x, self.W) + F.linear(x, self.B)
        return out

class MessageGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(MessageGraphConvolution, self).__init__()
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.B = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.B)

    @staticmethod
    def message(x, edge_index):
        """
        message step of the message passing algorithm for GCNs.

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :return: message vector with shape [num_nodes, num_in_features]. Messages correspond to the old node values.

        Hint: check out torch.Tensor.index_add function
        """
        num_nodes = x.size(0)
        num_features = x.size(1)

        messages = x[edge_index[0]] 
        aggregated_messages = torch.zeros(num_nodes, num_features, device=x.device)
        aggregated_messages.index_add_(0, edge_index[1], messages)
        sum_weight = torch.zeros(num_nodes, dtype=x.dtype, device=x.device)
        sum_weight.index_add_(0, edge_index[1], torch.ones(edge_index[1].size(0), dtype=x.dtype, device=x.device))
        sum_weight = torch.clamp(sum_weight, min=1.0)
        aggregated_messages = aggregated_messages / sum_weight.unsqueeze(1)

        return aggregated_messages

    def update(self, x, messages):
        """
        update step of the message passing algorithm for GCNs.

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param messages: messages vector with shape [num_nodes, num_in_features]
        :return: updated values of nodes. shape: [num_nodes, num_out_features]
        """
        x = F.linear(messages, self.W) + F.linear(x, self.B)
        return x

    def forward(self, x, edge_index):
        message = self.message(x, edge_index)
        x = self.update(x, message)
        return x


class GraphAttention(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttention, self).__init__()
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.a = nn.Parameter(torch.Tensor(out_features * 2))

        nn.init.xavier_uniform_(self.W)
        nn.init.uniform_(self.a, 0, 1)

    def forward(self, x, edge_index, debug=False):
        """
        Forward propagation for GATs.
        Follow the implementation of Graph attention networks (Veličković et al. 2018).

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :param debug: used for tests
        :return: updated values of nodes. shape: [num_nodes, num_out_features]
        :return: debug data for tests:
                 messages -> messages vector with shape [num_nodes, num_out_features], i.e. Wh from Veličković et al.
                 edge_weights_numerator -> unnormalized edge weightsm i.e. exp(e_ij) from Veličković et al.
                 softmax_denominator -> per destination softmax normalizer

        Hint: the GAT implementation uses only 1 parameter vector and edge index with self loops
        Hint: It is easier to use/calculate only the numerator of the softmax
              and weight with the denominator at the end.

        Hint: check out torch.Tensor.index_add function
        """
        edge_index, _ = add_self_loops(edge_index)

        sources, destinations = edge_index
        activations = F.linear(x, self.W)
        messages = activations[sources]

        attention_inputs = torch.cat([messages, activations[destinations]], dim=1)

        edge_weights_numerator = torch.exp(F.leaky_relu(
            (attention_inputs * self.a).sum(dim=1)
        ))
        weighted_messages = messages * edge_weights_numerator.unsqueeze(1)

        softmax_denominator = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        softmax_denominator.index_add_(0, destinations, edge_weights_numerator)

        aggregated_messages = torch.zeros(x.size(0), activations.size(1), dtype=x.dtype, device=x.device)
        aggregated_messages.index_add_(0, destinations, weighted_messages)
        aggregated_messages = aggregated_messages / softmax_denominator.unsqueeze(1)
        return aggregated_messages, {'edge_weights': edge_weights_numerator, 'softmax_weights': softmax_denominator,
                                     'messages': messages}

