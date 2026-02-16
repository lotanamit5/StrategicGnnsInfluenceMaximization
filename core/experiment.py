import torch
from argparse import ArgumentParser
from torch_geometric.data import Data
from functools import partial

from utils.general_helpers import set_seed, record_args
from utils.basic_classes import StrategicModelParameters, DataSet
from core.models import ExtendedSGConv, StrategicSGConv
from utils.train_or_test import train, test, line_search
from core.simulate_strategic_movement import simulate_strategic_movement


def expanded_get_attr(args: ArgumentParser, attr_name: str):
    """
    Expands the getattr method to include None if the attr is not present

    :param args: ArgumentParser
    :param attr_name: str
    """
    return getattr(args, attr_name) if hasattr(args, attr_name) else None


class Experiment(object):
    """
    Creates a single experiment which is later activated with the run function

    :param args: ArgumentParser
    """
    def __init__(self, args: ArgumentParser):
        get_arg = partial(expanded_get_attr, args)

        self.dataset_name = args.dataset_name
        self.num_layers = args.num_layers

        # args only for synthetic dataset
        self.alpha = get_arg(attr_name='alpha')
        self.test_iterations = get_arg(attr_name='test_iterations')

        # args only for real dataset
        self.temp = get_arg(attr_name='temp')
        self.lr = get_arg(attr_name='lr')
        self.epochs = get_arg(attr_name='epochs')
        self.decay = get_arg(attr_name='decay')

        # StrategicModelParameters
        self.strategic_model_parameters = \
            StrategicModelParameters(cost_lim=args.cost_lim, train_iterations=args.train_iterations,
                                     test_iterations=self.test_iterations,
                                     temp=self.temp)

        # result reproduction
        self.seed = args.seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # add prints
        print(f'######################## STARTING EXPERIMENT ########################', flush=True)
        self.name_of_run = record_args(args)

    def _train_per_dataset_type(self, data: Data, model: ExtendedSGConv) -> ExtendedSGConv:
        """
        Initiates training per dataset type

        :param data: Data
        :param model: ExtendedSGConv
        :return: model: ExtendedSGConv
        """
        if self.dataset_name is DataSet.SYNTHETIC:
            model = line_search(data=data, model=model)
        else:
            strategic_optimizer = torch.optim.Adam([dict(params=model.parameters(),
                                                         weight_decay=self.decay)], lr=self.lr)
            train(data=data, model=model, optimizer=strategic_optimizer,
                  epochs=self.epochs)
        return model

    def run(self, data: Data = None):
        """
        The only usage for this class
        """
        set_seed(seed=self.seed)
        if data is None:
            data = self.dataset_name.get_dataset(num_layers=self.num_layers).to(device=self.device)

        # basic model
        basic_model = ExtendedSGConv(in_channels=data.num_features, out_channels=1,
                                     K=self.num_layers, alpha=self.alpha).to(device=self.device)
        basic_model = self._train_per_dataset_type(data=data, model=basic_model)
        clean_accs = test(data=data, model=basic_model)

        # strategic movement
        tmp_x = data.x.clone()
        data.x = simulate_strategic_movement(x_init=data.x, edge_index=data.edge_index, model=basic_model,
                                             strategic_model_parameters=self.strategic_model_parameters,
                                             exact_movement=True)
        attacked_accs = test(data=data, model=basic_model)
        data.x = tmp_x

        # strategic model
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
        
        return {
                'train_clean': clean_accs[0],
                'train_naive': attacked_accs[0],
                'train_robust': robust_accs[0],
                'test_clean': clean_accs[1],
                'test_naive': attacked_accs[1],
                'test_robust': robust_accs[1]
        }
