import torch
import numpy as np
from tqdm import tqdm

class LTModel:
    def __init__(self, edge_index, num_nodes, weight_type='uniform', device='cpu'):
        self.device = device
        self.num_nodes = num_nodes
        self.weight_type = weight_type
        
        # Sort edges by TARGET (dst) for incoming edge lookup
        src, dst = edge_index[0], edge_index[1]
        sort_idx = torch.argsort(dst)
        self.src_sorted = src[sort_idx].to(device)
        self.dst_sorted = dst[sort_idx].to(device)
        
        # CSR Pointers
        self.degree = torch.bincount(self.dst_sorted, minlength=num_nodes)
        self.ptr = torch.cat([torch.tensor([0], device=device), 
                              torch.cumsum(self.degree, 0)])

        # Precompute Weights
        if weight_type == 'pagerank':
            self.weights = self._compute_pagerank_weights(self.src_sorted, self.dst_sorted, num_nodes)
        else:
            self.weights = None # Implicitly 1.0 everywhere

    def _compute_pagerank_weights(self, src, dst, num_nodes):
        # 1. Raw importance = Out-Degree of the influencer
        out_degree = torch.bincount(src, minlength=num_nodes).float()
        out_degree[out_degree == 0] = 1.0 
        edge_scores = out_degree[src]
        
        # 2. Normalize so sum(incoming) = 1.0
        sum_scores = torch.zeros(num_nodes, device=self.device)
        sum_scores.scatter_add_(0, dst, edge_scores)
        denominators = sum_scores[dst]
        denominators[denominators == 0] = 1.0
        
        return edge_scores / denominators

    def generate_rr_sets_vectorized(self, num_samples, max_depth=50, batch_size=10000):
        all_rr_sets = []
        pbar = tqdm(total=num_samples, desc="Generating RR-Sets", unit="walks")
        
        for start_idx in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - start_idx)
            
            # Initialize Walkers
            current_nodes = torch.randint(0, self.num_nodes, (current_batch_size,), device=self.device)
            traces = [current_nodes]
            active_mask = torch.ones(current_batch_size, dtype=torch.bool, device=self.device)
            
            for _ in range(max_depth):
                if not active_mask.any(): break
                
                # Get active walkers
                active_indices = torch.where(active_mask)[0]
                active_curr = current_nodes[active_indices]
                degs = self.degree[active_curr]
                
                # Kill walkers with no neighbors
                can_move = degs > 0
                if not can_move.any(): break
                
                # Filter down to moving walkers
                moving_indices = active_indices[can_move]
                moving_nodes = active_curr[can_move]
                moving_degs = degs[can_move]
                starts = self.ptr[moving_nodes]

                # --- SAMPLING STEP ---
                if self.weight_type == 'uniform':
                    # Fast Uniform Sampling
                    offsets = (torch.rand(len(moving_nodes), device=self.device) * moving_degs).long()
                    next_indices = starts + offsets
                else:
                    # Weighted Sampling (Padding Approach)
                    # 1. Pad weights to max degree in this batch
                    max_d = moving_degs.max().item()
                    batch_w = torch.zeros(len(moving_nodes), max_d, device=self.device)
                    
                    # 2. Fill the padded buffer (Slightly slow but correct)
                    # We have to loop or use sophisticated scatter/gather here. 
                    # For stability/readability, we use a loop over unique degrees or simple masking.
                    # Optimization: We treat the flattened weights as a 1D selection problem? 
                    # No, multinomial needs 2D [batch, categories].
                    
                    # FAST PATH: Construct a temporary 'dense' weight matrix for this batch
                    # This is heavy on VRAM if max_d is huge, but fine for CORA/Citation graphs.
                    for i, (st, d) in enumerate(zip(starts, moving_degs)):
                        batch_w[i, :d] = self.weights[st : st+d]
                        
                    # 3. Sample
                    # batch_w contains weights, 0 padding elsewhere. 
                    # torch.multinomial handles 0 probability automatically.
                    offsets = torch.multinomial(batch_w, 1).squeeze(1) # [Batch]
                    next_indices = starts + offsets

                # Advance
                next_nodes = self.src_sorted[next_indices]
                current_nodes[moving_indices] = next_nodes
                
                # Update Mask
                active_mask[:] = False
                active_mask[moving_indices] = True
                traces.append(current_nodes.clone())

            # Save Results
            traces_stacked = torch.stack(traces, dim=1).cpu().numpy()
            batch_rr_sets = [set(traces_stacked[i]) for i in range(current_batch_size)]
            all_rr_sets.extend(batch_rr_sets)
            pbar.update(current_batch_size)
            
        pbar.close()
        return all_rr_sets

    def select_seeds_greedy(self, rr_sets, k):
        seeds = []
        pbar = tqdm(range(k), desc="Greedy Selection", unit="seed")
        for _ in pbar:
            counts = {}
            for rr_set in rr_sets:
                for node in rr_set:
                    counts[node] = counts.get(node, 0) + 1
            if not counts: break
            best_node = max(counts, key=counts.get)
            seeds.append(best_node)
            rr_sets = [s for s in rr_sets if best_node not in s]
            pbar.set_postfix({"Marginal Gain": counts[best_node]})
        pbar.close()
        return seeds