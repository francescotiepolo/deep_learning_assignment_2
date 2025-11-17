import unittest
import unittest

import torch
import torch.optim as optim
from torch_geometric.utils import add_self_loops

from graph_cnn import MatrixGraphConvolution, MessageGraphConvolution, GraphAttention


class TestConvolutionLayers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.in_features = 3
        cls.out_features = 5
        cls.num_nodes = 4
        cls.edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])

    def setUp(self):
        self.input_data = torch.randn(self.num_nodes, self.in_features, requires_grad=True)

    def shared_logic_for_output_shape_test(self, layer_class):
        self.conv_layer = layer_class(self.in_features, self.out_features)
        output = self.conv_layer(self.input_data, self.edge_index)
        self.assertEqual(output.shape, (self.num_nodes, self.out_features))

    def shared_logic_for_test_gradient_flow(self, conv_layer):
        self.conv_layer = conv_layer(self.in_features, self.out_features)
        output = self.conv_layer(self.input_data, self.edge_index)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(self.input_data.grad)
        self.assertTrue(torch.all(torch.isfinite(self.input_data.grad)))

    def shared_logic_for_test_parameters_updated(self, conv_layer):
        self.conv_layer = conv_layer(self.in_features, self.out_features)
        self.optimizer = optim.SGD(self.conv_layer.parameters(), lr=0.01)

        output = self.conv_layer(self.input_data, self.edge_index)
        loss = output.sum()
        loss.backward()

        self.optimizer.step()
        for param in self.conv_layer.parameters():
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.any(param.grad != 0))

    def test_output_shape_matrix_GCN(self):
        self.shared_logic_for_output_shape_test(MatrixGraphConvolution)

    def test_output_shape_message_GCN(self):
        self.shared_logic_for_output_shape_test(MessageGraphConvolution)

    def test_output_shape_GAT(self):
        self.shared_logic_for_output_shape_test(GraphAttention)

    def test_gradient_flow_matrix_GCN(self):
        self.shared_logic_for_test_gradient_flow(MatrixGraphConvolution)

    def test_gradient_flow_message_GCN(self):
        self.shared_logic_for_test_gradient_flow(MessageGraphConvolution)

    def test_gradient_flow_GAT(self):
        self.shared_logic_for_test_gradient_flow(GraphAttention)

    def test_parameters_updated_matrix_GCN(self):
        self.shared_logic_for_test_gradient_flow(MatrixGraphConvolution)

    def test_parameters_updated_message_GCN(self):
        self.shared_logic_for_test_gradient_flow(MessageGraphConvolution)

    def test_parameters_updated_GAT(self):
        self.shared_logic_for_test_gradient_flow(GraphAttention)

    def test_implementation_consistency(self):
        conv_1 = MessageGraphConvolution(self.in_features, self.out_features)
        conv_2 = MatrixGraphConvolution(self.in_features, self.out_features)
        conv_2.W = conv_1.W
        conv_2.B = conv_1.B

        output1 = conv_1(self.input_data, self.edge_index)
        output2 = conv_2(self.input_data, self.edge_index)
        self.assertTrue(torch.allclose(output1, output2))

    def get_gat_aux(self):
        layer = GraphAttention(self.in_features, self.out_features)
        _, aux = layer(self.input_data, self.edge_index, debug=True)
        print(aux.keys())
        return aux

    def test_softmax_denominator_shape(self):
        softmax_denominator = self.get_gat_aux()['softmax_weights']
        self.assertEqual(softmax_denominator.shape, (self.num_nodes,))

    def test_edge_weights_shape(self):
        edge_weights = self.get_gat_aux()['edge_weights']
        self.assertEqual(edge_weights.shape, (self.edge_index.shape[1] + self.num_nodes,))

    def test_messages_shape(self):
        messages = self.get_gat_aux()['messages']
        self.assertEqual(messages.shape, (self.edge_index.shape[1] + self.num_nodes, self.out_features))

    def test_softmax(self):
        aux = self.get_gat_aux()
        edge_weights = aux['edge_weights']
        softmax_denominator = aux['softmax_weights']
        edge_index_with_loops, _ = add_self_loops(self.edge_index)
        sources, destinations = edge_index_with_loops
        alpha = edge_weights / softmax_denominator[destinations]

        per_node_alpha_sum = torch.zeros(self.num_nodes)
        per_node_alpha_sum = per_node_alpha_sum.index_add(0, destinations, alpha)
        self.assertTrue(torch.allclose(per_node_alpha_sum, torch.ones_like(per_node_alpha_sum)))

if __name__ == '__main__':
    unittest.main()
