from copy import deepcopy
import os
import pickle
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from .model import GEARS_Model
from .inference import evaluate, compute_metrics
from .utils import loss_fct, get_similarity_network, print_sys, GeneSimNetwork, \
                  create_cell_graph_dataset_for_prediction

torch.manual_seed(0)

import warnings
warnings.filterwarnings("ignore")

class GEARS:
    """
    GEARS base model class
    """

    def __init__(self, pert_data, datasets,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 weight_bias_track = False, 
                 proj_name = 'GEARS', 
                 exp_name = 'GEARS'):

        self.weight_bias_track = weight_bias_track
        
        if self.weight_bias_track:
            import wandb
            wandb.init(project=proj_name, name=exp_name)  
            self.wandb = wandb
        else:
            self.wandb = None
        
        self.device = device
        self.config = None
        
        self.dataloader = pert_data.dataloader
        self.adata = pert_data.adata
        self.node_map = pert_data.node_map
        self.data_path = '/data/data_gene_co'
        self.dataset_name = 'sciplex'
        self.seed = 1
        self.train_gene_set_size = 0.75
        self.set2conditions = pert_data.set2conditions
        self.gene_list = pert_data.gene_names.values.tolist()
        self.num_genes = len(self.gene_list)
        self.smiles = pert_data.canon_smiles_unique_sorted
        self.dosages = pert_data.dosages
        self.datasets = datasets
        self.saved_pred = {}
        self.saved_logvar_sum = {}
        
        self.ctrl_expression = torch.tensor(
            np.mean(self.adata.X[self.adata.obs.condition == 'control'],
                    axis=0)).reshape(-1, ).to(self.device)
    def tunable_parameters(self):
        """
        Return the tunable parameters of the model

        Returns
        -------
        dict
            Tunable parameters of the model

        """

        return {'hidden_size': 'hidden dimension, default 64',
                'num_go_gnn_layers': 'number of GNN layers for GO graph, default 1',
                'num_gene_gnn_layers': 'number of GNN layers for co-expression gene graph, default 1',
                'decoder_hidden_size': 'hidden dimension for gene-specific decoder, default 16',
                'num_similar_genes_go_graph': 'number of maximum similar K genes in the GO graph, default 20',
                'num_similar_genes_co_express_graph': 'number of maximum similar K genes in the co expression graph, default 20',
                'coexpress_threshold': 'pearson correlation threshold when constructing coexpression graph, default 0.4',
                'uncertainty': 'whether or not to turn on uncertainty mode, default False',
                'uncertainty_reg': 'regularization term to balance uncertainty loss and prediction loss, default 1',
                'direction_lambda': 'regularization term to balance direction loss and prediction loss, default 1'
               }
    
    def model_initialize(self, hidden_size = 64,
                         num_go_gnn_layers = 1, 
                         num_gene_gnn_layers = 1,
                         decoder_hidden_size = 16,
                         num_similar_genes_go_graph = 20,
                         num_similar_genes_co_express_graph = 20,                    
                         coexpress_threshold = 0.4,
                         uncertainty = False, 
                         uncertainty_reg = 1,
                         direction_lambda = 1e-1,
                         G_go = None,
                         G_go_weight = None,
                         G_coexpress = None,
                         G_coexpress_weight = None,
                         no_perturb = False,
                         **kwargs
                        ):
        
        self.config = {'hidden_size': hidden_size,
                       'num_go_gnn_layers' : num_go_gnn_layers, 
                       'num_gene_gnn_layers' : num_gene_gnn_layers,
                       'decoder_hidden_size' : decoder_hidden_size,
                       'num_similar_genes_go_graph' : num_similar_genes_go_graph,
                       'num_similar_genes_co_express_graph' : num_similar_genes_co_express_graph,
                       'coexpress_threshold': coexpress_threshold,
                       'uncertainty' : uncertainty, 
                       'uncertainty_reg' : uncertainty_reg,
                       'direction_lambda' : direction_lambda,
                       'G_go': G_go,
                       'G_go_weight': G_go_weight,
                       'G_coexpress': G_coexpress,
                       'G_coexpress_weight': G_coexpress_weight,
                       'device': self.device,
                       'num_genes': self.num_genes,
                       'gene_list': self.gene_list,
                    #    'num_perts': self.num_perts,
                       'no_perturb': no_perturb,
                       'smiles' : self.smiles,
                       'dosages' : self.dosages,
                      }
        
        if self.wandb:
            self.wandb.config.update(self.config)

        if self.config['G_coexpress'] is None:
            ## calculating co expression similarity graph
            edge_list = get_similarity_network(network_type='co-express',
                                               adata=self.adata,
                                               threshold=coexpress_threshold,
                                               k=num_similar_genes_co_express_graph,
                                               data_path=self.data_path,
                                               data_name=self.dataset_name,
                                               seed=self.seed,
                                               train_gene_set_size=self.train_gene_set_size,
                                               set2conditions=self.set2conditions)

            sim_network = GeneSimNetwork(edge_list, self.gene_list, node_map = self.node_map)
            self.config['G_coexpress'] = sim_network.edge_index
            self.config['G_coexpress_weight'] = sim_network.edge_weight

        self.model = GEARS_Model(self.config).to(self.device)
        self.best_model = deepcopy(self.model)
        
    def load_pretrained(self, path):
        state_dict = torch.load(os.path.join(path, 'pretrained_model.pt'), map_location = torch.device('cpu'))
    
        self.model = state_dict.to(self.device)
        self.best_model = self.model
    
    def save_model(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        
        if self.config is None:
            raise ValueError('No model is initialized...')
        
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
       
        torch.save(self.best_model.state_dict(), os.path.join(path, 'model.pt'))
    
    def predict(self, pert_list):
        
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'control']
        for pert in pert_list:
            for i in pert:
                if i not in self.pert_list:
                    raise ValueError(i+ " is not in the perturbation graph. "
                                        "Please select from GEARS.pert_list!")
        
        if self.config['uncertainty']:
            results_logvar = {}
            
        self.best_model = self.best_model.to(self.device)
        self.best_model.eval()
        results_pred = {}
        results_logvar_sum = {}
        
        from torch_geometric.data import DataLoader
        for pert in pert_list:
            try:
                #If prediction is already saved, then skip inference
                results_pred['_'.join(pert)] = self.saved_pred['_'.join(pert)]
                if self.config['uncertainty']:
                    results_logvar_sum['_'.join(pert)] = self.saved_logvar_sum['_'.join(pert)]
                continue
            except:
                pass
            
            cg = create_cell_graph_dataset_for_prediction(pert, self.ctrl_adata,
                                                    self.pert_list, self.device)
            loader = DataLoader(cg, 300, shuffle = False)
            batch = next(iter(loader))
            batch.to(self.device)

            with torch.no_grad():
                if self.config['uncertainty']:
                    p, unc = self.best_model(batch)
                    results_logvar['_'.join(pert)] = np.mean(unc.detach().cpu().numpy(), axis = 0)
                    results_logvar_sum['_'.join(pert)] = np.exp(-np.mean(results_logvar['_'.join(pert)]))
                else:
                    p = self.best_model(batch)
                    
            results_pred['_'.join(pert)] = np.mean(p.detach().cpu().numpy(), axis = 0)
                
        self.saved_pred.update(results_pred)
        
        if self.config['uncertainty']:
            self.saved_logvar_sum.update(results_logvar_sum)
            return results_pred, results_logvar_sum
        else:
            return results_pred

    
    
    def train(self, epochs = 1, 
              lr = 1e-4,
              weight_decay = 5e-4
             ):
        
        train_loader = self.dataloader['train_loader']
        val_loader = self.dataloader['val_loader']
        test_loader = self.dataloader['test_loader']
        best_model = deepcopy(self.model)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay = weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=1)
        total_loss = 0.0

        min_val = np.inf
        print_sys('Start Training...')
        for epoch in range(epochs):
            self.model.train()
            num_batches = len(train_loader)
            for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}", leave=True):
                batch = batch.to(self.device)
                optimizer.zero_grad()
                y = batch.y
                pred = self.model(batch)
                loss = loss_fct(pred, y, batch.pert)
                loss.backward()
                
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                optimizer.step()

                if self.wandb:
                    self.wandb.log({'training_loss': loss.item()})
                total_loss += loss.item()

            scheduler.step()
            total_loss = total_loss / num_batches          
            print('Epoch {} Train Loss: {:.6f}'.format(epoch + 1, total_loss))

            self.model.eval()
            with torch.no_grad():
                print('done')
                train_res = evaluate(train_loader, self.model,
                                    self.config['uncertainty'], self.device)
                
                val_res = evaluate(val_loader, self.model,
                                    self.config['uncertainty'], self.device)

            train_metrics, _ = compute_metrics(train_res)
            val_metrics, _ = compute_metrics(val_res)

            metrics = ['mse', 'pearson', 'r2']
            for m in metrics:
                print({'train_' + m: train_metrics[m],
                            'val_'+m: val_metrics[m],
                            'train_de_' + m: train_metrics[m + '_de'],
                            'val_de_'+m: val_metrics[m + '_de']})
               
            if val_metrics['mse_de'] < min_val:
                min_val = val_metrics['mse_de']
                best_model = deepcopy(self.model)
        
        print_sys("Done!")
        self.best_model = best_model

             
        print_sys("Start Testing...")
        with torch.no_grad():
            test_res = evaluate(test_loader, self.best_model,
                                self.config['uncertainty'], self.device)
        
        test_metrics, _ = compute_metrics(test_res)
        metrics = ['mse', 'pearson', 'r2']
        for m in metrics:
            print({'test_' + m: test_metrics[m],
                        'test_de_' + m: test_metrics[m + '_de']})
        print_sys('Done!')
        
        return test_res

