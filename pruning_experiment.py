import pandas as pd
import numpy as np
import torch

from core.experiment import Experiment
from core.models import ExtendedSGConv, StrategicSGConv
from core.simulate_strategic_movement import simulate_strategic_movement
from utils.general_helpers import set_seed
from utils.train_or_test import test
from utils.record_trial import record
from influence_maximization import LTModel


class PruningExperiment(Experiment):
    def __init__(self, args):
        super().__init__(args)
        self.k = args.k_prune
        self.method = args.method
        self.args = args
    
    def run(self):
        set_seed(seed=self.seed)
        data = self.dataset_name.get_dataset(num_layers=self.num_layers).to(device=self.device)
        
        # prune
        if self.k > 0:
            print(f"Pruning edges with k={self.k} using method '{self.method}'...", flush=True)
            im_model = LTModel(
                edge_index=data.edge_index, 
                num_nodes=data.num_nodes, 
                weight_type=self.method, # 'pagerank', 'uniform'
                device=self.device
            )
            num_samples = min(200_000, data.num_nodes * 5)
            rr_sets = im_model.generate_rr_sets_vectorized(num_samples=num_samples)
            K = int(self.k * data.num_nodes)
            seeds = im_model.select_seeds_greedy(rr_sets, k=K)
            src_np = data.edge_index[0].cpu().numpy()
            seeds_np = np.array(seeds)
            mask_np = np.isin(src_np, seeds_np)
            keep_mask = torch.tensor(~mask_np, device=self.device, dtype=torch.bool)
            data.edge_index = data.edge_index[:, keep_mask]
        
        # basic model
        print("Training basic model on pruned graph...", flush=True)
        basic_model = ExtendedSGConv(in_channels=data.num_features, out_channels=1,
                                    K=self.num_layers, alpha=self.alpha).to(device=self.device)
        basic_model = self._train_per_dataset_type(data=data, model=basic_model)
        clean_accs = test(data=data, model=basic_model)

        # strategic movement
        print("Simulating strategic movement and testing basic model...", flush=True)
        tmp_x = data.x.clone()
        data.x = simulate_strategic_movement(x_init=data.x, edge_index=data.edge_index, model=basic_model,
                                            strategic_model_parameters=self.strategic_model_parameters,
                                            exact_movement=True)
        attacked_accs = test(data=data, model=basic_model)
        data.x = tmp_x
                
        # strategic model
        print("Training strategic model on pruned graph...", flush=True)
        set_seed(seed=self.seed)
        strategic_model = StrategicSGConv(in_channels=data.num_features, out_channels=1, K=self.num_layers,
                                        strategic_model_parameters=self.strategic_model_parameters,
                                        alpha=self.alpha).to(device=self.device)
        strategic_model = self._train_per_dataset_type(data=data, model=strategic_model)
        robust_accs = test(data=data, model=strategic_model)

        # results summary
        print('Non-strategic -- Train: {:.4f}, Test: {:.4f}'
            .format(*clean_accs), flush=True)
        print('Naive         -- Train: {:.4f}, Test: {:.4f}'
            .format(*attacked_accs), flush=True)
        print('Robust        -- Train: {:.4f}, Test: {:.4f}'
            .format(*robust_accs), flush=True)
            
        record(
            exp_name=self.args.exp_name,
            args=self.args,
            metrics={
                'clean_train': clean_accs[0],
                'clean_test': clean_accs[1],
                'naive_train': attacked_accs[0],
                'naive_test': attacked_accs[1],
                'robust_train': robust_accs[0],
                'robust_test': robust_accs[1],
                'k_prune': self.k
            }
        )
        
        
