import torch
import pandas as pd
from tqdm import tqdm

from core.arguments import get_synthetic_dataset_parser, get_real_datasets_parser
from core.experiment import Experiment
from utils.basic_classes import DataSet
from influence_maximization import LTModel

def sort_nodes(data, condition):
    # Calculate degrees (out-degree based on source in edge_index)
    # edge_index is [2, E], edge_index[0] is source
    device = data.x.device
    # Rankings
    # Degree: Sort nodes by degree descending
    if condition == 'degree':
        src, _ = data.edge_index
        deg = torch.bincount(src, minlength=data.num_nodes)
        sorted_nodes = torch.argsort(deg, descending=True)
    if condition == 'random':    
        sorted_nodes = torch.randperm(data.num_nodes, device=device)
    if condition == 'im':
        im_model = LTModel(
            edge_index=data.edge_index, 
            num_nodes=data.num_nodes, 
            weight_type='pagerank'
        )
        num_samples = min(200_000, data.num_nodes * 5)
        rr_sets = im_model.generate_rr_sets_vectorized(num_samples=num_samples)
        seeds = im_model.select_seeds_greedy(rr_sets, k=data.num_nodes)
        not_seeds = set(range(data.num_nodes)) - set(seeds)
        sorted_nodes = torch.tensor(seeds + list(not_seeds), device=device)
    return sorted_nodes


all_results = []
num_layers = 1
dataset = DataSet.SYNTHETIC
data = dataset.get_dataset(num_layers)
device = data.x.device
for condition in ['degree', 'random', 'im']:
    print(f"Running pruning experiment for condition: {condition}")
    sorted_nodes = sort_nodes(data=data, condition=condition)
    
    for portion_keep in tqdm([0, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
        portion_remove = 100 - portion_keep
        num_nodes_remove = int((portion_remove / 100) * data.num_nodes)
        nodes_to_remove = sorted_nodes[:num_nodes_remove]
        
        # prune
        src, dst = data.edge_index
        disconnect_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        disconnect_mask[nodes_to_remove] = True
        edge_mask = ~disconnect_mask[src]
        pruned_edge_index = data.edge_index[:, edge_mask]
        
        data_clone = data.clone()
        data_clone.edge_index = pruned_edge_index
        
        parser = get_real_datasets_parser()
        args = parser.parse_args()
        setattr(args, 'dataset_name', dataset)
        setattr(args, 'num_layers', num_layers)
        results = Experiment(args).run(data=data_clone)
        
        results['p'] = portion_keep
        results['condition'] = condition
        all_results.append(results)
        
all_results = pd.DataFrame(all_results)
path = f"pruning_experiment_results_{dataset}_dst.csv"
all_results.to_csv(path, index=False)
print(f"Experiment completed and results saved to {path}")
        

        