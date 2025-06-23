# Graph Construction Utilities for Brain Connectivity Networks
# Handles loading, processing, and visualization of functional connectivity matrices

import os
import scipy.io as sio
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce
import dataload as Reader
from torch_geometric.utils import to_networkx
from opt import *
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt

opt = OptInit().initialize()

data_folder = opt.data_path

def read_sigle_data(data):
    # Convert functional connectivity matrix to PyTorch Geometric graph format
    # Input: numpy array of connectivity matrix
    # Output: PyTorch Geometric Data object with nodes, edges, and attributes
    # Refer to the data download process: https://github.com/SamitHuang/EV_GCN and https://github.com/xxlya/BrainGNN_Pytorch.

    # Take absolute values and create NetworkX graph
    pcorr = np.abs(data)
    num_nodes = pcorr.shape[0]
    G = nx.from_numpy_array(pcorr)
    A = nx.adjacency_matrix(G)
    adj = A.tocoo()
    
    # Extract edge attributes from connectivity matrix
    edge_att = np.zeros(len(adj.row))
    for i in range(len(adj.row)):
        edge_att[i] = pcorr[adj.row[i], adj.col[i]]
    
    # Create edge index and remove self-loops
    edge_index = np.stack([adj.row, adj.col])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes, num_nodes)
    
    # Prepare node features
    att = data
    att[att== float('inf')] = 0  # Replace infinite values
    
    # Apply threshold to keep only strong connections
    kind = np.where(edge_att > opt.alpha)[0]
    edge_index=edge_index[:,kind]
    edge_att=edge_att[kind]
    
    # Create PyTorch Geometric Data object
    att_torch = torch.from_numpy(att).float()
    graph = Data(x=att_torch, edge_index=edge_index.long(), edge_attr=edge_att)
    
    # ------------------------------------------------------------
    # Pre-compute hemisphere sub-graphs so they don't need to be
    # recomputed inside every forward pass on the GPU.
    # ------------------------------------------------------------
    left_mask  = (graph.edge_index[0] < 50) & (graph.edge_index[1] < 50)
    right_mask = (graph.edge_index[0] >= 50) & (graph.edge_index[1] >= 50)

    graph.left_edge_index = graph.edge_index[:, left_mask]
    graph.left_edge_attr  = graph.edge_attr[left_mask]

    graph.right_edge_index = graph.edge_index[:, right_mask]
    graph.right_edge_attr  = graph.edge_attr[right_mask]
    
    return graph

def visualize_subject_graph(subject_id, kind='fc_matrix', variable='arr_0', save_plot=True):
    # Visualize brain connectivity graph for a specific subject
    # Creates 4-panel plot: correlation matrix, adjacency matrix, and two network layouts
    print(f"Visualizing graph for subject: {subject_id}")
    
    # Load subject's connectivity data
    fl = os.path.join(data_folder, subject_id, 
                     subject_id + "_run-01_" + kind + ".npz")
    
    if not os.path.exists(fl):
        print(f"ERROR: File not found - {fl}")
        return None
        
    try:
        loaded = np.load(fl)
        
        if variable not in loaded.files:
            print(f"ERROR: Variable '{variable}' not found in {fl}")
            print(f"Available variables: {loaded.files}")
            return None
            
        matrix = loaded[variable]
        print(f"Loaded matrix shape: {matrix.shape}")
        
        # Apply Fisher's z-transformation (arctanh normalization)
        with np.errstate(divide='ignore', invalid='ignore'):
            norm_matrix = np.arctanh(matrix)
        norm_matrix = np.nan_to_num(norm_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create graph and convert to NetworkX for visualization
        graph = read_sigle_data(norm_matrix)
        networkx_graph = to_networkx(graph, to_undirected=True)
        
        print(f"Graph statistics:")
        print(f"  - Number of nodes: {networkx_graph.number_of_nodes()}")
        print(f"  - Number of edges: {networkx_graph.number_of_edges()}")
        print(f"  - Graph density: {nx.density(networkx_graph):.4f}")
        
        # Create 4-panel visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Panel 1: Original correlation matrix heatmap
        im1 = ax1.imshow(matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax1.set_title(f'Original Correlation Matrix\n{subject_id}')
        ax1.set_xlabel('Brain Region')
        ax1.set_ylabel('Brain Region')
        plt.colorbar(im1, ax=ax1)
        
        # Panel 2: Thresholded adjacency matrix
        num_nodes = graph.x.shape[0]
        adj_matrix = np.zeros((num_nodes, num_nodes))
        edge_index = graph.edge_index.numpy()
        edge_attr = graph.edge_attr.numpy() if graph.edge_attr is not None else np.ones(edge_index.shape[1])
        
        # Build symmetric adjacency matrix from edges
        for i in range(edge_index.shape[1]):
            adj_matrix[edge_index[0, i], edge_index[1, i]] = edge_attr[i]
            adj_matrix[edge_index[1, i], edge_index[0, i]] = edge_attr[i]  # Make symmetric
        
        im2 = ax2.imshow(adj_matrix, cmap='Blues')
        ax2.set_title(f'Thresholded Adjacency Matrix\n(α={opt.alpha})')
        ax2.set_xlabel('Brain Region')
        ax2.set_ylabel('Brain Region')
        plt.colorbar(im2, ax=ax2)
        
        # Panel 3: Network graph (circular layout)
        pos = nx.circular_layout(networkx_graph)
        nx.draw_networkx_nodes(networkx_graph, pos, ax=ax3, node_color='lightblue', 
                              node_size=50, alpha=0.8)
        nx.draw_networkx_edges(networkx_graph, pos, ax=ax3, edge_color='gray', 
                              alpha=0.5, width=0.5)
        ax3.set_title('Network Graph (Circular Layout)')
        ax3.axis('off')
        
        # Panel 4: Network graph (spring layout)
        pos_spring = nx.spring_layout(networkx_graph, k=1, iterations=50)
        nx.draw_networkx_nodes(networkx_graph, pos_spring, ax=ax4, node_color='lightcoral', 
                              node_size=50, alpha=0.8)
        nx.draw_networkx_edges(networkx_graph, pos_spring, ax=ax4, edge_color='gray', 
                              alpha=0.5, width=0.5)
        ax4.set_title('Network Graph (Spring Layout)')
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plot:
            filename = f'{subject_id}_connectivity_graph.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved as: {filename}")
        
        plt.show()
        
        # Calculate and print network topology metrics
        if networkx_graph.number_of_nodes() > 0:
            try:
                avg_clustering = nx.average_clustering(networkx_graph)
                print(f"  - Average clustering coefficient: {avg_clustering:.4f}")
            except:
                print("  - Could not compute clustering coefficient")
                
            try:
                if nx.is_connected(networkx_graph):
                    avg_path_length = nx.average_shortest_path_length(networkx_graph)
                    print(f"  - Average shortest path length: {avg_path_length:.4f}")
                else:
                    print("  - Graph is not connected")
            except:
                print("  - Could not compute path length")
        
        return graph, networkx_graph
        
    except Exception as e:
        print(f"ERROR: Could not visualize {subject_id} - {str(e)}")
        return None

def get_networks(subject_list, kind, atlas_name="ho", variable='arr_0'):
    # Load and process connectivity matrices for multiple subjects
    # Returns: list of processed graphs and list of successfully loaded subjects
    graphs = []
    successful_subjects = []
    skipped_subjects = []
    
    print(f"Attempting to load {len(subject_list)} subjects...")
    
    for subject in subject_list:
        # Construct file path for connectivity matrix
        fl = os.path.join(data_folder, subject, 
                         subject + "_run-01_" + kind + ".npz")
        
        # File existence check
        if not os.path.exists(fl):
            print(f"SKIPPED: File not found - {fl}")
            skipped_subjects.append(f"{subject} (file not found)")
            continue
            
        # File size check
        if os.path.getsize(fl) == 0:
            print(f"SKIPPED: Empty file - {fl}")
            skipped_subjects.append(f"{subject} (empty file)")
            continue
            
        try:
            loaded = np.load(fl)
            
            # Variable existence check
            if variable not in loaded.files:
                print(f"SKIPPED: Variable '{variable}' not found in {fl}")
                skipped_subjects.append(f"{subject} (missing variable '{variable}')")
                continue
                
            matrix = loaded[variable]
            
            # Matrix validity checks
            if matrix.size == 0:
                print(f"SKIPPED: Empty matrix in {fl}")
                skipped_subjects.append(f"{subject} (empty matrix)")
                continue
                
            if np.all(matrix == 0):
                print(f"SKIPPED: All-zero matrix in {fl}")
                skipped_subjects.append(f"{subject} (all-zero matrix)")
                continue
                
            if not np.isfinite(matrix).all():
                print(f"WARNING: Non-finite values in {fl}, attempting to clean...")
                matrix = np.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Apply Fisher's z-transformation for normalization
            with np.errstate(divide='ignore', invalid='ignore'):
                norm_matrix = np.arctanh(matrix)
            norm_matrix = np.nan_to_num(norm_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Create graph object
            graph = read_sigle_data(norm_matrix)
            
            # Validate graph has edges after thresholding
            if graph.edge_index.shape[1] == 0:
                print(f"SKIPPED: No edges after thresholding for {subject}")
                skipped_subjects.append(f"{subject} (no edges after threshold)")
                continue
            
            # Add adjacency matrix to graph (required by Brain_connectomic_graph model)
            a_graph = to_networkx(graph)
            A = np.array(nx.adjacency_matrix(a_graph).todense())
            graph.adj = A
            
            graphs.append(graph)
            successful_subjects.append(subject)
            
        except Exception as e:
            print(f"SKIPPED: Error processing {subject} - {str(e)}")
            skipped_subjects.append(f"{subject} (processing error: {str(e)})")
            continue
    
    # Print loading summary
    print(f"\nLoading Summary:")
    print(f"  Successfully loaded: {len(successful_subjects)} subjects")
    print(f"  Skipped: {len(skipped_subjects)} subjects")
    
    if len(skipped_subjects) > 0 and len(skipped_subjects) <= 10:
        print("  Skipped subjects:")
        for subject in skipped_subjects:
            print(f"    - {subject}")
    elif len(skipped_subjects) > 10:
        print(f"  First 10 skipped subjects:")
        for subject in skipped_subjects[:10]:
            print(f"    - {subject}")
        print(f"    ... and {len(skipped_subjects) - 10} more")
    
    return graphs, successful_subjects

def get_node_feature():
    # Main function to load all subject connectivity graphs
    # Returns: (graphs, successful_subjects) tuple
    
    subject_list = Reader.get_ids()  # Get list of all subject IDs
    print(f"Total subjects in list: {len(subject_list)}")
    
    # Load functional connectivity matrices
    graphs, successful_subjects = get_networks(subject_list, kind='fc_matrix')
    
    print(f"Final result: {len(graphs)} graphs loaded for {len(successful_subjects)} subjects")
    
    return graphs, successful_subjects

if __name__ == "__main__":
    # Test visualization with the first subject
    subject_ids = Reader.get_ids()
    test_subject = subject_ids[0] if len(subject_ids) > 0 else "sub-002S0295"
    
    print(f"Testing graph visualization with subject: {test_subject}")
    result = visualize_subject_graph(test_subject)
    
    if result is not None:
        print("Visualization completed successfully!")
    else:
        print("Visualization failed!")
    
    # Also run the original functionality
    print("\n" + "="*50)
    print("Running original functionality...")
    graphs, subjects = get_node_feature()
    print(f"Final result: Loaded {len(graphs)} graphs for {len(subjects)} subjects.")
    