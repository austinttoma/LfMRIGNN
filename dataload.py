from sklearn.model_selection import StratifiedKFold
from graph import get_node_feature
import csv
import numpy as np
import torch
from torch import nn
import sys
from opt import *

opt = OptInit().initialize()

def standardization_intensity_normalization(dataset, dtype):
    # Apply z-score normalization: (x - mean) / std
    mean = dataset.mean()
    std  = dataset.std()
    return ((dataset - mean) / std).astype(dtype)

def intensityNormalisationFeatureScaling(dataset, dtype):
    # Apply min-max normalization: (x - min) / (max - min)
    max = dataset.max()
    min = dataset.min()
    return ((dataset - min) / (max - min)).astype(dtype)

class dataloader():
    # Main data loading and preprocessing class for brain connectivity analysis
    
    def __init__(self): 
        self.pd_dict = {}
        self.num_classes = opt.num_classes

    def load_data(self):
        # Load and preprocess brain connectivity data with demographic information
        # Returns: (raw_features, labels, phenotypic_data, phenotypic_scores)
        # Load graph features and get list of successfully processed subjects
        self.raw_features, successful_subjects = get_node_feature()
        print(f"\nDataloader: Using {len(successful_subjects)} successfully loaded subjects")
        
        # Get demographic and clinical data for valid subjects only
        labels = get_subject_score(successful_subjects, score='Group')
        ages = get_subject_score(successful_subjects, score='Age')
        genders = get_subject_score(successful_subjects, score='Sex')
        
        num_nodes = len(successful_subjects)
        
        # Initialize arrays for labels and demographics
        y_onehot = np.zeros([num_nodes, self.num_classes])
        y = np.zeros([num_nodes])
        age = np.zeros([num_nodes], dtype=np.float32)
        gender = np.zeros([num_nodes], dtype=int)
        
        # Map Alzheimer's disease labels to numeric values
        label_map = {'CN': 0, 'MCI': 1, 'AD': 2}  # Cognitive Normal, Mild Cognitive Impairment, Alzheimer's Disease
        
        # Process each subject's data
        for i in range(num_nodes):
            subject_id = successful_subjects[i]
            
            # Convert string labels to numeric
            label_str = labels[subject_id]
            if label_str in label_map:
                label_num = label_map[label_str]
            else:
                print(f"Warning: Unknown label '{label_str}' for subject {subject_id}")
                label_num = 0  # Default to CN if unknown
                
            # Handle binary vs multi-class classification
            if self.num_classes == 2:
                # Binary: CN vs (MCI + AD)
                binary_label = 0 if label_num == 0 else 1
                y_onehot[i, binary_label] = 1
                y[i] = binary_label
            else:
                # Multi-class: CN vs MCI vs AD
                y_onehot[i, label_num] = 1
                y[i] = label_num
            
            # Process demographic data
            age[i] = float(ages[subject_id])
            gender[i] = 1 if genders[subject_id] == 'M' else 0  # M=1, F=0

        # Store for later use
        self.successful_subjects = successful_subjects
        self.y = y 
        
        # Create phenotypic data matrix (gender, age)
        phonetic_data = np.zeros([num_nodes, 2], dtype=np.float32)
        phonetic_data[:,0] = gender
        phonetic_data[:,1] = age
        
        # Store in dictionary format for graph construction
        self.pd_dict['Sex'] = np.copy(phonetic_data[:, 0])
        self.pd_dict['Age'] = np.copy(phonetic_data[:,1])

        phonetic_score = self.pd_dict
        
        print(f"Dataloader: Final dataset has {len(self.raw_features)} graphs and {len(self.y)} labels")
        return self.raw_features, self.y, phonetic_data, phonetic_score
    

    def data_split(self, n_folds):
        # Generate stratified k-fold cross-validation splits
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=666)
        cv_splits = list(skf.split(self.raw_features, self.y))
        return cv_splits 

    def get_inputs(self, nonimg, embeddings, phonetic_score):
        # Construct heterogeneous population graph (HPG) edges based on subject similarity
        # Returns: (same_index, diff_index) - edges for same-gender and different-gender connections
        # Create gender-based adjacency mask
        S = self.create_type_mask()
        self.node_ftr = np.array(embeddings.detach().cpu().numpy())
        n = self.node_ftr.shape[0]
        
        # Pre-allocate edge arrays
        num_edge = n * (1 + n) // 2 - n  # Number of edges in complete graph (no self-loops)
        edge_index = np.zeros([2, num_edge], dtype=np.int64)
        aff_score = np.zeros(num_edge, dtype=np.float32)
        
        # Get phenotypic similarity matrix
        aff_adj = get_static_affinity_adj(phonetic_score)
        
        # Build edge list with similarity scores
        flatten_ind = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_index[:, flatten_ind] = [i, j]
                aff_score[flatten_ind] = aff_adj[i][j]
                flatten_ind += 1
        assert flatten_ind == num_edge, "Error in computing edge input"

        # Keep only edges above threshold (sparsification)
        keep_ind = np.where(aff_score > opt.beta)[0]
        edge_index = edge_index[:, keep_ind]
        
        # Separate edges by gender relationship
        same_row = []    # Same-gender connections
        same_col = []
        diff_row = []    # Different-gender connections  
        diff_col = []
        
        for i in range(edge_index.shape[1]):
            if S[edge_index[0, i], edge_index[1, i]] == 1:
                same_row.append(edge_index[0, i])
                same_col.append(edge_index[1, i])
            else:
                diff_row.append(edge_index[0, i])
                diff_col.append(edge_index[1, i])

        same_index = np.stack((same_row, same_col)).astype(np.int64)
        diff_index = np.stack((diff_row, diff_col)).astype(np.int64)

        return same_index, diff_index

    def create_type_mask(self):
        # Create binary mask indicating same-gender vs different-gender subject pairs
        subject_list = self.successful_subjects
        num_nodes = len(subject_list)
        type_matrix = np.zeros((num_nodes, num_nodes), dtype=np.int64)

        type = get_subject_score(subject_list, score='Sex')
        
        # Set 1 for same gender pairs, 0 for different gender
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if type[subject_list[i]] == type[subject_list[j]]:
                    type_matrix[i, j] = 1
                    type_matrix[j, i] = 1

        type_matrix = torch.from_numpy(type_matrix)
        device='cuda:0'
        # return type_matrix.to(device) # For CUDA GPUs
        return type_matrix.to("cpu") # Force CPU usage

def get_subject_score(subject_list, score):
    # Load specific phenotypic scores for given subjects from CSV file
    scores_dict = {}
    phenotype = opt.phenotype_path
    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['SUB_ID'] in subject_list:
                scores_dict[row['SUB_ID']] = row[score]
    return scores_dict

def get_ids(subject_file=None, num_subjects=None):
    # Load subject IDs from file with optional limiting
    path = subject_file if subject_file else opt.subject_IDs_path
    subject_IDs = np.genfromtxt(path, dtype=str, ndmin=1)
    if num_subjects is not None:
        num_subjects = int(num_subjects)
        return subject_IDs[:num_subjects]
    return subject_IDs


def create_affinity_graph_from_scores(scores, pd_dict):
    # Create similarity graph based on phenotypic features
    # Categorical features (like gender): exact match gets score +1
    # Continuous features (like age): similar values (within threshold) get score +1
    num_nodes = len(pd_dict[scores[0]]) 
    graph = np.zeros((num_nodes, num_nodes))

    for l in scores:
        label_dict = pd_dict[l]

        if l in ['Age']:
            # Age similarity: subjects within 2 years get connection
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[k]) - float(label_dict[j]))
                        if val < 2:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # Handle missing values
                        pass

        else:
            # Categorical similarity: exact match gets connection
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_dict[k] == label_dict[j]:
                        graph[k, j] += 1
                        graph[j, k] += 1

    return graph

def get_static_affinity_adj(pd_dict):
    # Compute normalized affinity matrix based on gender similarity
    # Create similarity graph based on gender
    pd_affinity = create_affinity_graph_from_scores(['Sex'], pd_dict)
    # Normalize for stable training
    pd_affinity = (pd_affinity - pd_affinity.mean(axis=0)) / pd_affinity.std(axis=0)
    return pd_affinity

class LabelSmoothingLoss(nn.Module):
    # Label smoothing loss to prevent overconfident predictions
    
    def __init__(self, smoothing=0.1, reduction='mean'):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, output, target):
        # Apply label smoothing and compute KL divergence loss
        n_classes = output.size(1)
        target_one_hot = torch.zeros_like(output).scatter(1, target.unsqueeze(1), 1)
        target_smooth = target_one_hot * (1 - self.smoothing) + (1 - target_one_hot) * self.smoothing / (n_classes - 1)
        log_probs = nn.functional.log_softmax(output, dim=1)
        loss = nn.functional.kl_div(log_probs, target_smooth, reduction='none').sum(1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class Logger(object):
    # Simple logger that writes to both console and file
    
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        # Write message to both console and log file
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass