# -*- coding: utf-8 -*-

import argparse

def build_args(model_name):
    
    parser = argparse.ArgumentParser(description=model_name)
    register_default_args(parser)
    args = parser.parse_args()

    return args

def register_default_args(parser): 
    
    # general settings
    parser.add_argument('--random_seed', type=int, default=123)    
    parser.add_argument("--dataset", type=str, default="cora", required=False,
                        help="The input dataset.")
    
    # settings for the genetic algorithm
    parser.add_argument('--num_individuals', type=int, default=20,
                        help='the population size')
    parser.add_argument('--num_individuals_param', type=int, default=4,
                        help='the population size')   
    parser.add_argument('--num_generations', type=int, default=50,
                        help='number of evolving generations')
    parser.add_argument('--num_generations_param', type=int, default=5,
                        help='number of evolving generations') 
    parser.add_argument('--num_parents', type=int, default=10,
                        help='number of parents for crossover')
    parser.add_argument('--num_parents_param', type=int, default=4,
                        help='number of parents for crossover')
    parser.add_argument('--num_offsprings', type=int, default=4, # number of parents pair for crossover
                        help='number of offsprings')
    parser.add_argument('--num_offsprings_param', type=int, default=2,
                        help='number of offsprings')
    
    parser.add_argument('--mutate_prob', type=float, default=0.02,
                        help='mutation probability')
    parser.add_argument('--super_ratio', type=float, default=0.6,
                        help=' probability')
    
    
    # settings for the gnn model
    parser.add_argument('--num_gnn_layers', type=int, default=2,
                        help='number of the GNN layers')
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")    
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")    
    parser.add_argument("--in-drop", type=float, default=0.6,
                        help="input feature dropout")
    parser.add_argument("--cuda", type=bool, default=True,
                        help="cuda")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu number")
    
    # parameter sharing
    parser.add_argument("--shared_params", action='store_true', default=False,
                        help="whether to share parameters")
    parser.add_argument("--half_epochs", action='store_true', default=False)
    parser.add_argument("--combined_evolution", action='store_true', default=False)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--param_initial_rate', type=float, default=0,
                        help='0 means all hyperparameters are initialized randomly')
#     parser.add_argument('--save_epoch', type=int, default=2)
#     parser.add_argument('--max_save_num', type=int, default=5)
#     parser.add_argument("--residual", action="store_false",
#                         help="use residual connection")
    
    
    
               