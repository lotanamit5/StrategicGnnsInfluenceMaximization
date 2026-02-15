from datetime import date
from utils.basic_classes import DataSet
from core.arguments import get_synthetic_dataset_parser
from pruning_experiment import PruningExperiment


if __name__ == '__main__':
    parser = get_synthetic_dataset_parser()
    parser.add_argument('-e', '--exp_name', type=str, default=date.today().isoformat(),
                        help='Name for the experiment (used for saving results)')
    parser.add_argument('-k', '--k_prune', type=float, default=0.0, 
                        help='Pruning parameter k (0 to 1)')
    parser.add_argument('-m', '--method', type=str, default='uniform', 
                        help='Pruning method ("uniform", "pagerank")')    
    args = parser.parse_args()

    # constant args
    setattr(args, 'dataset_name', DataSet.SYNTHETIC)
    setattr(args, 'num_layers', 1)
    PruningExperiment(args).run()
