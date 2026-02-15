import torch
from sklearn.svm import SVC

class EdgePruner:
    def __init__(self, data, 
                 method: str='hadamard', 
                 seed: int=None):
        if method not in ['hadamard', 'concat', 'absdiff']:
            raise ValueError(f"Method must be 'hadamard', 'concat' or 'absdiff', got {method}")
        self.seed = seed
        self.method = method
        self.clf = SVC(probability=True, random_state=seed)
        
        X, Y = self._create_dataset(data, balanced=True)
        X_np = X.detach().cpu().numpy()
        Y_np = Y.detach().cpu().numpy()
        
        self.clf.fit(X_np, Y_np)

    def _create_dataset(self, data, balanced: bool = False):
        edge_index = data.edge_index
        x = data.x
        y = data.y
        
        if x is None or edge_index is None or y is None:
             raise ValueError("Data object must contain 'x', 'y', and 'edge_index'")

        row, col = edge_index
        x1 = x[row]
        x2 = x[col]
        y1 = y[row]
        y2 = y[col]
        
        Y_prime = (y1 == y2).long()
        
        if self.method == 'concat':
            X_prime = torch.cat([x1, x2], dim=1)
        elif self.method == 'hadamard':
            X_prime = x1 * x2
        elif self.method == 'absdiff':
            X_prime = torch.abs(x1 - x2)
            
        if balanced:
            indices_0 = (Y_prime == 0).nonzero(as_tuple=True)[0]
            indices_1 = (Y_prime == 1).nonzero(as_tuple=True)[0]
            
            min_count = min(len(indices_0), len(indices_1))
            
            g = torch.Generator()
            if self.seed is not None:
                g.manual_seed(self.seed)
            else:
                g.seed()
                
            idx_0 = indices_0[torch.randperm(len(indices_0), generator=g)[:min_count]]
            idx_1 = indices_1[torch.randperm(len(indices_1), generator=g)[:min_count]]
            
            indices = torch.cat([idx_0, idx_1])
            indices = indices[torch.randperm(len(indices), generator=g)]
            
            X_prime = X_prime[indices]
            Y_prime = Y_prime[indices]
            
        return X_prime, Y_prime

    def prune(self, data, k: float):
        if not (0 <= k <= 1):
             raise ValueError("k must be in [0, 1]")
        
        # Create copy of the data
        data_copy = data.clone()
        
        # Convert to features
        X, _ = self._create_dataset(data_copy)
        X_np = X.detach().cpu().numpy()
        
        # Predict probabilities
        probs = self.clf.predict_proba(X_np)[:, 1]
        
        num_edges = data_copy.edge_index.size(1)
        num_remove = int(num_edges * k)
        
        if num_remove > 0:
            probs_tensor = torch.from_numpy(probs).to(data_copy.edge_index.device)
            _, top_indices = torch.topk(probs_tensor, num_remove)
            
            mask = torch.ones(num_edges, dtype=torch.bool, device=data_copy.edge_index.device)
            mask[top_indices] = False
            
            data_copy.edge_index = data_copy.edge_index[:, mask]
            
        return data_copy
