import numpy as np
import os.path as osp
import time
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from gnn import GraphNet
import gc
from thop import profile

import warnings
warnings.filterwarnings('ignore')

def evaluate(output, labels, mask):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    return correct.item() * 1.0 / mask.sum().item()

class GNNModelManager(object):
    
    def __init__(self, args):
        
        self.args = args
        self.loss_fn = torch.nn.functional.nll_loss
        self.device = torch.device("cuda" if self.args.cuda else "cpu")
        
    
    def load_data(self, dataset='Citeseer'):
        
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
        dataset = Planetoid(path, dataset, split='public', transform=T.NormalizeFeatures())
        data = dataset[0]
        
#         print(np.sum(np.array(data.val_mask), 0))
        
#         data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
#         data.train_mask[:-1000] = 1
#         data.val_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
#         data.val_mask[data.num_nodes - 1000: data.num_nodes - 500] = 1
#         data.test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
#         data.test_mask[data.num_nodes - 500:] = 1
        
        self.data = data
        
        
#         print(data.edge_index)
#         print(data.edge_index.shape)
        
        self.args.num_class = data.y.max().item() + 1
        self.args.in_feats = self.data.num_features
    
    def load_param(self):
        # don't share param
        pass    

    def update_args(self, args):
        self.args = args

    def save_param(self, model, update_all=False):
        pass
        
    def build_gnn(self, actions, drop_outs, shared_params):
        
        model = GraphNet(self.args, self.args.num_gnn_layers,
                         actions, self.args.in_feats, self.args.num_class, shared_params=shared_params,
                         drop_outs=drop_outs, multi_label=False,
                         batch_normal=False, residual=False)
        return model
   
    # train from scratch
    def train(self, actions, params, shared_params):
        # change the last gnn dimension to num_class
        actions[-1] = self.args.num_class
        print('==================================\ncurrent training actions={}, params={}'.format(actions, params))
        
        # create gnn model
        learning_rate = params[-2]
        weight_decay = params[-1]
        drop_outs = params[:-2]
        
        gnn_model = self.build_gnn(actions, drop_outs, shared_params)
        gnn_model.to(self.device)
        
        # define optimizer
        optimizer = torch.optim.Adam(gnn_model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)
        
        # run model to get accuracy
        model, val_acc, test_acc, flopparam, times = self.run_model(gnn_model, 
                                        optimizer, 
                                        self.loss_fn, 
                                        self.data, 
                                        self.args.epochs,
                                        early_stop=self.args.early_stopping,
                                        return_best=True,
                                        show_info=False)

        return val_acc, test_acc, gnn_model.shared_params, flopparam, times
        
    def run_model(self, model, optimizer, loss_fn, data, epochs, early_stop=5, 
                  return_best=False, cuda=True, need_early_stop=False, show_info=False):
        torch.cuda.empty_cache()
        gc.collect()
        
        dur = []
        begin_time = time.time()
        best_performance, best_test = 0, 0
        min_val_loss = float("inf")
        min_train_loss = float("inf")
        model_val_acc = 0
        model_test_acc = 0
        counter = 0 # early stop counter
        
        if self.args.half_epochs and sum(model.loaded)==self.args.num_gnn_layers:
            epochs = int(epochs / 2)
            print('half epochs, epochs={}'.format(epochs))
        
#         print("Number of train datas:", data.train_mask.sum())
        for epoch in range(1, epochs + 1):
#             print(data.edge_index.shape, data.x.shape, data.y.shape)
            data.x, data.y, data.edge_index = data.x.to(self.device), data.y.to(self.device), data.edge_index.to(self.device)
            data.train_mask, data.val_mask, data.test_mask = data.train_mask.to(self.device), data.val_mask.to(self.device), data.test_mask.to(self.device)
            if epoch == 1:
                t0 = time.time()
            model.train()
            logits = model(data.x, data.edge_index)
            logits = F.log_softmax(logits, 1)
            
            loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            if epoch == 1:
                train_time = time.time() - t0

            # evaluate
            if epoch == 1:
                t0 = time.time()
            model.eval()
            logits = model(data.x, data.edge_index)
            logits = F.log_softmax(logits, 1)
            if epoch == 1:
                inference_time = time.time() - t0
                flops, params = profile(model, inputs=(data.x, data.edge_index), verbose=False)
            
            train_acc = evaluate(logits, data.y, data.train_mask)
            val_acc = evaluate(logits, data.y, data.val_mask)
            test_acc = evaluate(logits, data.y, data.test_mask)

            loss = loss_fn(logits[data.val_mask], data.y[data.val_mask])
            val_loss = loss.item()
            if val_loss < min_val_loss:  # and train_loss < min_train_loss
                min_val_loss = val_loss
                min_train_loss = train_loss
                model_val_acc = val_acc
                model_test_acc = test_acc
                counter = 0
                if val_acc > best_performance:
                    best_performance = val_acc
                    best_test = test_acc
            else:
                counter += 1
                if early_stop>0 and counter==early_stop:
                    break
            if show_info:
                time_used = time.time() - begin_time
                print(
                    "Epoch {:05d} | Loss {:.4f} | acc {:.4f} | val_acc {:.4f} | test_acc {:.4f} | time {}".format(
                        epoch, loss.item(), train_acc, val_acc, test_acc, time_used))

        print("val_score:{:.4f}, test_score:{:.4f}, consumed_time:{:.2f}".format(model_val_acc, model_test_acc, time.time() - begin_time), '\n')
        if return_best:
            return model, best_performance, best_test, [flops, params], [train_time, inference_time]
        else:
            return model, model_val_acc, model_test_acc, [flops, params], [train_time, inference_time]