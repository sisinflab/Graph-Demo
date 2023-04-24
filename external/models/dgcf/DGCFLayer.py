from abc import ABC

import torch
from torch_geometric.nn import MessagePassing


class DGCFLayer(MessagePassing, ABC):
    def __init__(self, intents):
        super(DGCFLayer, self).__init__(aggr='add', node_dim=-3)
        self.intents = intents

    def forward(self, x, edge_index, edge_index_intents):
        normalized_edge_index_intents = torch.softmax(edge_index_intents, dim=0)
        deg_inv_sqrt_row = []
        deg_inv_sqrt_col = []
        for i in range(self.intents):
            A_i_tensor = torch.sparse_coo_tensor(edge_index, normalized_edge_index_intents[i], (x.shape[0], x.shape[0]))
            deg_inv_sqrt_row.append(1 / torch.sqrt(torch.sparse.sum(A_i_tensor, dim=0).to_dense()))
            deg_inv_sqrt_col.append(1 / torch.sqrt(torch.sparse.sum(A_i_tensor, dim=1).to_dense()))
        row, col = edge_index
        deg_inv_sqrt_row = torch.stack(deg_inv_sqrt_row, dim=0)
        deg_inv_sqrt_col = torch.stack(deg_inv_sqrt_col, dim=0)
        deg_inv_sqrt_row[deg_inv_sqrt_row == float('inf')] = 0
        deg_inv_sqrt_col[deg_inv_sqrt_col == float('inf')] = 0
        norm = deg_inv_sqrt_row[:, row] * deg_inv_sqrt_col[:, col] * normalized_edge_index_intents
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return torch.unsqueeze(norm.permute(1, 0), -1) * x_j
