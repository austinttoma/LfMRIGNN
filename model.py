# FC-HGNN Model Architecture
# Combines brain connectomic graphs with heterogeneous population graphs

from torch_geometric.nn import ChebConv,TransformerConv
from dataload import dataloader
from opt import *
import torch.nn.functional as F
from torch import nn
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import subgraph
from torch_geometric.nn.dense.diff_pool import dense_diff_pool
from torch_geometric.nn import SAGPooling
import numpy as np
import os

opt = OptInit().initialize()


class Brain_connectomic_graph(torch.nn.Module):
    # Individual brain connectivity graph processing module
    # Handles intra-hemispheric and inter-hemispheric convolutions with dual-channel pooling

    def __init__(self):
        super(Brain_connectomic_graph, self).__init__()
        self._setup()
        
    def _setup(self):
        # Left and right hemisphere GCN layers (first level)
        self.graph_convolution_l_1 = GCNConv(100,64)
        self.graph_convolution_r_1 = GCNConv(100,64)

        # Left and right hemisphere GCN layers (second level)
        self.graph_convolution_l_2 = GCNConv(64,20)
        self.graph_convolution_r_2 = GCNConv(64,20)

        # Inter-hemispheric global convolution
        self.graph_convolution_g_1 = GCNConv(20,20)

        # Dual-channel pooling layers
        self.pooling_1 = SAGPooling(20, opt.k1)  # Channel 1: SAG pooling
        self.socre_gcn = ChebConv(20, int(opt.k2*100), K=3, normalization='sym')  # Channel 2: Chebyshev conv
        self.pooling_2= dense_diff_pool

        # Cross-channel convolution parameters
        self.weight = nn.Parameter(torch.FloatTensor(64, 20)).to(opt.device)
        self.bns=nn.BatchNorm1d(20).to(opt.device)
        nn.init.xavier_normal_(self.weight)

    def forward(self, data):
        # Extract graph components
        edges, features = data.edge_index, data.x
        edges, features = edges.to(opt.device), features.to(opt.device)

        edge_attr = data.edge_attr
        edge_attr = edge_attr.to(opt.device).to(torch.float32)

        adj=data.adj
        adj=torch.tensor(adj)
        adj=adj.float()
        adj = adj.to(opt.device)

        # Define brain hemisphere regions (100 regions total)
        leftBrain = torch.tensor([i for i in range(50)], dtype=torch.float).to(opt.device)  # Regions 0-49
        rightBrain = torch.tensor([i for i in range(50, 100)], dtype=torch.float).to(opt.device)  # Regions 50-99

        # Create hemisphere-specific subgraphs
        new_left_edges,new_left_edge_attr = subgraph(subset=leftBrain.type(torch.long),edge_index=edges,edge_attr=edge_attr)
        new_right_edges,new_right_edge_attr = subgraph(subset=rightBrain.type(torch.long), edge_index=edges, edge_attr=edge_attr)

        # First level intra-hemispheric convolutions
        features = F.dropout(features, p=opt.dropout, training=self.training)
        node_features_left = torch.nn.functional.leaky_relu(self.graph_convolution_l_1(features, new_left_edges, new_left_edge_attr))
        node_features_right = torch.nn.functional.leaky_relu(self.graph_convolution_r_1(features, new_right_edges, new_right_edge_attr))
        
        # Combine left and right hemisphere features
        node_features_1 = torch.zeros(100,64).to(opt.device)
        node_features_1[leftBrain.long(),:] = node_features_left[leftBrain.long(),:]
        node_features_1[rightBrain.long(), :] = node_features_right[rightBrain.long(), :]

        # Second level intra-hemispheric convolutions
        node_features_1 = F.dropout(node_features_1, p=opt.dropout, training=self.training)
        node_features_left = torch.nn.functional.leaky_relu(self.graph_convolution_l_2(node_features_1, new_left_edges, new_left_edge_attr))
        node_features_right = torch.nn.functional.leaky_relu(self.graph_convolution_r_2(node_features_1, new_right_edges, new_right_edge_attr))
        node_features_2 = torch.zeros(100,20).to(opt.device)
        node_features_2[leftBrain.long(),:] = node_features_left[leftBrain.long(),:]
        node_features_2[rightBrain.long(), :] = node_features_right[rightBrain.long(), :]

        # Inter-hemispheric convolution (global brain connectivity)
        node_features_2 = torch.nn.functional.leaky_relu(self.graph_convolution_g_1(node_features_2, edges, edge_attr))

        # Local-Global Dual-channel Pooling (LGP)
        # Channel 1: SAG pooling
        pooling_features, edges, edge_attr,batch, perm, score = self.pooling_1(node_features_2, edges,edge_attr)

        # Channel 2: Dense differentiable pooling
        ass_matrix=F.softmax(self.socre_gcn(node_features_2,edges),dim=-1)
        H_coarse,assign_matrix, link_loss, ent_loss = self.pooling_2(node_features_2, adj,ass_matrix)

        # Cross-channel convolution to combine dual-channel outputs
        pooled_size = len(perm)  # Number of nodes after pooling
        inter_channel_adj = features.new_zeros(pooled_size, int(opt.k2*100))
        assign_matrix = torch.squeeze(ass_matrix)
        j = 0
        # Map pooled nodes to assignment matrix
        for i in range(0, 100):
            if i in perm:
                inter_channel_adj[j, :] = assign_matrix[i, :]
                j = j + 1

        # Final feature combination and graph embedding
        H_coarse = torch.squeeze(H_coarse)
        H1 = torch.matmul(inter_channel_adj, H_coarse)
        H2 = pooling_features + H1
        graph_embedding = H2.view(1, -1)

        return graph_embedding

class HPG(nn.Module):
    # Heterogeneous Population Graph module
    # Processes population-level relationships using transformer convolutions
    
    def __init__(self, input_dim=1800):
        super(HPG, self).__init__()
        self.num_layers = 4
        self.convs1 = nn.ModuleList()  # Same-gender connections
        self.convs2 = nn.ModuleList()  # Different-gender connections
        self.bns = nn.ModuleList()
        
        # Build transformer convolution layers
        self.convs1.append(TransformerConv(in_channels=input_dim, out_channels=20, heads=1))
        self.convs2.append(TransformerConv(in_channels=input_dim, out_channels=20, heads=1))
        self.bns.append(nn.BatchNorm1d(20))
        
        self.convs1.append(TransformerConv(in_channels=20, out_channels=20, heads=1))
        self.convs2.append(TransformerConv(in_channels=20, out_channels=20, heads=1))
        self.bns.append(nn.BatchNorm1d(20))
        self.convs1.append(TransformerConv(in_channels=20, out_channels=20, heads=1))
        self.convs2.append(TransformerConv(in_channels=20, out_channels=20, heads=1))
        self.bns.append(nn.BatchNorm1d(20))
        self.convs1.append(TransformerConv(in_channels=20, out_channels=20, heads=1))
        self.convs2.append(TransformerConv(in_channels=20, out_channels=20, heads=1))
        self.bns.append(nn.BatchNorm1d(20))
        
        # Final classification layer
        self.out_fc = nn.Linear(80, opt.num_classes)
        
        # Learnable fusion weights for same/different gender features
        self.weights1 = torch.nn.Parameter(torch.empty(4).fill_(0.8))  # Same-gender weights
        self.weights2 = torch.nn.Parameter(torch.empty(4).fill_(0.2))  # Different-gender weights

        self.a = torch.nn.Parameter(torch.Tensor(20, 1))

    def reset_parameters(self):
        # Reset all layer parameters
        for conv in self.convs1:
            conv.reset_parameters()
        for conv in self.convs2:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.out_fc.reset_parameters()
        self.a.reset_parameters()
        torch.nn.init.normal_(self.weights)

    def forward(self, features, same_index,diff_index):
        x = features

        # Layer 1: Process same-gender and different-gender connections separately
        x = F.dropout(x, p=opt.dropout, training=self.training)
        x1 = self.convs1[0](x,same_index)      # Same-gender features
        x2 = self.convs2[0](x,diff_index)     # Different-gender features
        
        # Adaptive fusion of same/different gender features
        weight1 = self.weights1[0] / (self.weights1[0]+ self.weights2[0])
        weight2 = self.weights2[0] / (self.weights1[0] + self.weights2[0])
        x=weight1*x1 + weight2*x2
        x = self.bns[0](x)
        x = F.leaky_relu(x, inplace=True)
        fc = x

        # Layer 2
        x = F.dropout(x, p=opt.dropout, training=self.training)
        x1 = self.convs1[1](x, same_index)
        x2 = self.convs2[1](x, diff_index)
        weight1 = self.weights1[1] / (self.weights1[1] + self.weights2[1])
        weight2 = self.weights2[1] / (self.weights1[1] + self.weights2[1])
        x = weight1 * x1 + weight2 * x2
        x = self.bns[1](x)
        x = F.leaky_relu(x, inplace=True)
        fc = torch.cat((fc, x), dim=-1)  # Concatenate features

        # Layer 3
        x = F.dropout(x, p=opt.dropout, training=self.training)
        x1 = self.convs1[2](x, same_index)
        x2 = self.convs2[2](x, diff_index)
        weight1 = self.weights1[2] / (self.weights1[2] + self.weights2[2])
        weight2 = self.weights2[2] / (self.weights1[2] + self.weights2[2])
        x = weight1 * x1 + weight2 * x2
        x = self.bns[2](x)
        x = F.leaky_relu(x, inplace=True)
        fc = torch.cat((fc, x), dim=-1)

        # Layer 4
        x = F.dropout(x, p=opt.dropout, training=self.training)
        x1 = self.convs1[3](x, same_index)
        x2 = self.convs2[3](x, diff_index)
        weight1 = self.weights1[3] / (self.weights1[3] + self.weights2[3])
        weight2 = self.weights2[3] / (self.weights1[3] + self.weights2[3])
        x = weight1 * x1 + weight2 * x2
        x = self.bns[3](x)
        x = F.leaky_relu(x, inplace=True)
        fc = torch.cat((fc, x), dim=-1)

        # Final classification
        x = self.out_fc(fc)

        return x

class fc_hgnn(torch.nn.Module):
    # Main FC-HGNN model combining individual brain graphs and population graph
    
    def __init__(self, nonimg, phonetic_score, dataloader_instance):
        super(fc_hgnn, self).__init__()
        self.nonimg = nonimg
        self.phonetic_score = phonetic_score
        self.dataloader = dataloader_instance
        self._setup()

    def _setup(self):
        # Initialize individual brain graph and population graph modules
        self.individual_graph_model = Brain_connectomic_graph()
        self.population_graph_model = HPG()

    def forward(self, graphs):
        embeddings = []

        # Process each individual brain connectivity graph
        for graph in graphs:
            embedding = self.individual_graph_model(graph)
            embeddings.append(embedding)
        embeddings = torch.cat(tuple(embeddings))

        # Build heterogeneous population graph edges
        same_index, diff_index = self.dataloader.get_inputs(self.nonimg, embeddings, self.phonetic_score)
        same_index = torch.tensor(same_index, dtype=torch.long).to(opt.device)
        diff_index = torch.tensor(diff_index, dtype=torch.long).to(opt.device)

        # Process population graph for final predictions
        predictions = self.population_graph_model(embeddings, same_index, diff_index)

        return predictions

class Graph_Transformer(nn.Module):
    # Graph transformer block with multi-head attention and feed-forward network
    
    def __init__(self, input_dim, output_num,head_num, hidden_dim):
        super(Graph_Transformer, self).__init__()
        # Multi-head self-attention layer
        self.graph_conv = TransformerConv(input_dim, output_num, head_num)
        self.lin_out = nn.Linear(input_dim, input_dim)

        # Feed-forward network layers
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, input_dim)
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        # Multi-head self-attention with residual connection
        out1 = self.lin_out(self.graph_conv(x, edge_index))

        # Feed-forward network with residual connections and layer normalization
        out2 = self.ln1(out1 + x)
        out3 = self.lin2(self.act(self.lin1(out2)))
        out4 = self.ln2(out3 + out2)

        return out4

# Debug: Check sample connectivity matrix dimensions
sample_files = [
    "./data/FC_Matrices/sub-002S0295/sub-002S0295_run-01_fc_matrix.npz",
    "./data/FC_Matrices/sub-002S0413/sub-002S0413_run-01_fc_matrix.npz"
]

for file_path in sample_files:
    if os.path.exists(file_path):
        data = np.load(file_path)
        matrix = data['arr_0']  # or whatever the key is
        print(f"File: {file_path}")
        print(f"Matrix shape: {matrix.shape}")
        print(f"Data keys: {list(data.keys())}")
        print("---")
