import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import os

from core.experiment import Experiment
from core.models import ExtendedSGConv, StrategicSGConv
from core.simulate_strategic_movement import simulate_strategic_movement
from influence_maximization import LTModel
from utils.basic_classes import DataSet
from utils.general_helpers import set_seed
from utils.train_or_test import test, train
from utils.record_trial import record

class CentralityExperiment(Experiment):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.dataset_name_str = args.dataset_name
        
    def run(self):
        set_seed(seed=self.seed)
        # Load dataset
        data = self.dataset_name.get_dataset(num_layers=self.num_layers).to(device=self.device)
        
        # Calculate degrees (out-degree based on source in edge_index)
        # edge_index is [2, E], edge_index[0] is source
        src, _ = data.edge_index
        deg = torch.bincount(src, minlength=data.num_nodes)
        
        # Rankings
        # Degree: Sort nodes by degree descending
        degree_sorted_nodes = torch.argsort(deg, descending=True)
        
        # Random: Shuffle nodes
        random_sorted_nodes = torch.randperm(data.num_nodes, device=self.device)
        
        # IM Pruning
        im_model = LTModel(
            edge_index=data.edge_index, 
            num_nodes=data.num_nodes, 
            weight_type='uniform', # 'pagerank', 'uniform'
        )
        num_samples = min(200_000, data.num_nodes * 5)
        rr_sets = im_model.generate_rr_sets_vectorized(num_samples=num_samples)
        seeds = im_model.select_seeds_greedy(rr_sets, k=data.num_nodes) # Get full ranking
        not_seeds = set(range(data.num_nodes)) - set(seeds)
        im_sorted_nodes = torch.tensor(seeds + list(not_seeds), device=self.device)
        
        results = []
        
        # Percentiles of nodes to disconnect (q%)
        # q corresponds to top-q% ranked nodes to disconnect
        # x-axis in plot is "portion of nodes that remain connected" (100 - q)
        # We vary remaining portion p from 10 to 100
        portions_remaining = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        # Conditions
        conditions = {
            'degree': degree_sorted_nodes,
            'random': random_sorted_nodes,
            'im': im_sorted_nodes
        }
        
        for p in portions_remaining:
            q = 100 - p
            num_nodes_to_disconnect = int(data.num_nodes * q / 100)
            
            print(f"--- Portion Remaining: {p}% (Disconnect top {q}%) ---")
            
            for cond_name, ranked_nodes in conditions.items():
                print(f"Condition: {cond_name}")
                
                # Identify nodes to disconnect
                if num_nodes_to_disconnect == 0:
                    nodes_to_disconnect = torch.tensor([], device=self.device)
                else:
                    nodes_to_disconnect = ranked_nodes[:num_nodes_to_disconnect]
                # Create mask of edges to keep
                # Remove edges where source is in nodes_to_disconnect
                if len(nodes_to_disconnect) > 0:
                    src, _ = data.edge_index
                    # Create a boolean mask for nodes
                    disconnect_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=self.device)
                    disconnect_mask[nodes_to_disconnect] = True
                    
                    # Keep edge if source is NOT in disconnect_mask
                    edge_mask = ~disconnect_mask[src]
                    pruned_edge_index = data.edge_index[:, edge_mask]
                else:
                    pruned_edge_index = data.edge_index.clone()
                
                # Clone data for experiments
                data_pruned = data.clone()
                data_pruned.edge_index = pruned_edge_index
                
                # --- Benchmark (Non-strategic) ---
                # Train Naive/Basic model on pruned graph
                # Test on pruned graph (NO movement)
                set_seed(self.seed) 
                benchmark_model = ExtendedSGConv(in_channels=data.num_features, out_channels=1,
                                                K=self.num_layers, alpha=self.alpha).to(device=self.device)
                benchmark_model = self._train_per_dataset_type(data=data_pruned, model=benchmark_model)
                benchmark_accs = test(data=data_pruned, model=benchmark_model)
                benchmark_test_acc = benchmark_accs[1]
                
                # --- Naive Approach ---
                # Train Basic model on pruned graph (anticipates no movement)
                # Test on pruned graph + Strategic Movement
                # Note: We reuse the trained benchmark_model as the "Naive" model (trained on static data)
                
                # Simulate movement against the Naive model
                tmp_x = data_pruned.x.clone()
                x_moved_naive = simulate_strategic_movement(
                    x_init=tmp_x, 
                    edge_index=data_pruned.edge_index, 
                    model=benchmark_model,
                    strategic_model_parameters=self.strategic_model_parameters,
                    exact_movement=True
                )
                
                data_pruned.x = x_moved_naive
                naive_accs = test(data=data_pruned, model=benchmark_model)
                naive_test_acc = naive_accs[1]
                
                # Calculate movement stats for Naive
                naive_moved_mask = (x_moved_naive != tmp_x).any(dim=1)
                naive_pct_moved = naive_moved_mask.float().mean().item() * 100
                
                # Calculate crossed stats for Naive
                logits_naive = benchmark_model(x_moved_naive, data_pruned.edge_index).to(device=self.device)
                naive_crossed_mask = naive_moved_mask & (logits_naive.flatten() >= 0)
                
                naive_pct_crossed = naive_crossed_mask.float().sum().item() / data.num_nodes * 100 
                
                data_pruned.x = tmp_x # Reset X
                
                # --- Robust Approach ---
                # Train Strategic model on pruned graph
                # Test on pruned graph + Strategic Movement
                set_seed(self.seed)
                robust_model = StrategicSGConv(in_channels=data.num_features, out_channels=1, K=self.num_layers,
                                            strategic_model_parameters=self.strategic_model_parameters,
                                            alpha=self.alpha).to(device=self.device)
                robust_model = self._train_per_dataset_type(data=data_pruned, model=robust_model)
                
                # Simulate movement explicitly for stats
                robust_x_moved = simulate_strategic_movement(
                     x_init=tmp_x,
                     edge_index=data_pruned.edge_index,
                     model=robust_model,
                     strategic_model_parameters=self.strategic_model_parameters,
                     exact_movement=True
                )
                
                # Calculate movement stats for Robust
                robust_moved_mask = (robust_x_moved != tmp_x).any(dim=1)
                robust_pct_moved = robust_moved_mask.float().mean().item() * 100
                
                # Classification on moved data (without re-moving)
                logits_robust = robust_model.non_strategic_forward(robust_x_moved, data_pruned.edge_index)
                
                robust_crossed_mask = robust_moved_mask & (logits_robust.flatten() >= 0).to(device=self.device)
                robust_pct_crossed = robust_crossed_mask.float().sum().item() / data.num_nodes * 100

                # Accuracy (using model internal movement)
                data_pruned.x = tmp_x
                robust_accs = test(data=data_pruned, model=robust_model)
                robust_test_acc = robust_accs[1]
                
                # Store results
                results.append({
                    'portion_remaining': p,
                    'condition': cond_name,
                    'benchmark_acc': benchmark_test_acc,
                    'naive_acc': naive_test_acc,
                    'robust_acc': robust_test_acc,
                    'naive_pct_moved': naive_pct_moved,
                    'naive_pct_crossed': naive_pct_crossed,
                    'robust_pct_moved': robust_pct_moved,
                    'robust_pct_crossed': robust_pct_crossed
                })

        # Save results
        df = pd.DataFrame(results)
        filename = f"results_centrality_{self.dataset_name_str}.csv"
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

if __name__ == '__main__':
    from core.arguments import get_real_datasets_parser
    parser = get_real_datasets_parser()
    args = parser.parse_args()
    setattr(args, 'dataset_name', DataSet.SYNTHETIC) 
    
    exp = CentralityExperiment(args)
    exp.run()
