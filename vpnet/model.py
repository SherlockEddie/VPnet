import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import logging
from .embedding import get_chemical_representation
from torch_geometric.nn import SGConv
from collections import OrderedDict
import numpy as np

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))
    
class VNN_cell(nn.Module): # BioVNN
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 biovnn_dict, 
                 act_func='Mish',
                 use_sigmoid_output=True, # True
                 dropout_p=0.5, # 0
                 only_combine_child_gene_group=True, #  True
                 neuron_min=10, 
                 neuron_ratio=0.2,
                 use_classification=True, # True
                 child_map_fully=None,  # None
                 for_lr_finder=True): #  False
        
        super(VNN_cell, self).__init__() 
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layer_names =biovnn_dict["community_hierarchy_dicts_all"]
        self.group_level_dict = biovnn_dict['community_level_dict']
        self.level_group_dict= biovnn_dict["level_community_dict"]
        
        self.use_sigmoid_output = use_sigmoid_output
        self.dropout_p = dropout_p
        self.only_combine_child_gene_group = only_combine_child_gene_group
        print("only_combine_child_gene_group: ",self.only_combine_child_gene_group)
        self.use_classification = use_classification
        self.child_map_fully = child_map_fully
        self.gene_group_idx = self.layer_names['gene_group_idx'] 
        self.idx_name = self.layer_names['idx_name']
        self.gene_feat = 1

        self.level_neuron_ct = dict()
        self.com_layers = nn.ModuleDict()
        self.bn_layers = nn.ModuleDict()
        self.output_layers = nn.ModuleDict()
        if self.dropout_p > 0:
            self.dropout_layers = nn.ModuleDict()
        self._set_layer_names()
        self.build_order = []
        self.child_map = biovnn_dict["mask"] 
        self.neuron_min = neuron_min
        self.neuron_ratio = neuron_ratio
        self._set_layers()
        if act_func.lower() == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func.lower() == 'mish':
            self.act_func = Mish()

        self.sigmoid = nn.Sigmoid()
        self.output = [None] * len(self.build_order)

        if self.only_combine_child_gene_group:
            logging.info("{} gene groups do not combine gene features".format(len(self.only_combine_gene_group_dict)))
        self.for_lr_finder = for_lr_finder
        
        self.norm = nn.LayerNorm(541, eps=1e-6)

    def _set_layers(self):
        neuron_n_dict = self._build_layers()
        if self.child_map_fully is not None:
            logging.info("Non-fully connected:")
        self.report_parameter_n()
        logging.debug(self.build_order)

    def _set_layer_names(self):
        for g in self.gene_group_idx.keys(): 
            self.com_layers[g] = None
            self.bn_layers['bn_{}'.format(g)] = None
            self.output_layers['output_{}'.format(g)] = None
            if self.dropout_p > 0:
                self.dropout_layers['drop_{}'.format(g)] = None

    def _build_layers(self, neuron_n_dict=None):
        self.child_dim = {}
        self.child = {}
        self.gene_child = {}
        neuron_to_build = list(range(len(self.child_map)))
        self.only_combine_gene_group_dict = {}
        if neuron_n_dict is None:
            neuron_n_dict = dict()
        while len(neuron_to_build) > 0:
            for i in neuron_to_build:
                j = i + self.input_dim
                children = self.child_map[i] 
                child_feat = [z for z in children if z < self.input_dim] 
                child_com = [self.idx_name[z] for z in children if z >= self.input_dim] 
                child_none = [self.com_layers[z] for z in child_com if self.com_layers[z] is None]
                if len(child_none) > 0:
                    logging.debug("Pass Gene group {} with {} children".format(j, len(children)))
                    continue
                neuron_name = self.idx_name[j] 
                self.child[neuron_name] = child_com
                self.gene_child[neuron_name] = child_feat

                if self.only_combine_child_gene_group and len(child_com) > 0:
                    children_n = len(child_com) 
                    if i == len(self.child_map) - 1: 
                        children_n = 512 
                    child_feat = []
                    self.only_combine_gene_group_dict[neuron_name] = 1
                else:
                    children_n = len(children)
                    if i == len(self.child_map) - 1: 
                        children_n += 512 

                logging.debug("Building gene group {} with {} children".format(j, len(children)))
                if i not in neuron_n_dict:
                    neuron_n = np.max([self.neuron_min, int(children_n * self.neuron_ratio)])
                    neuron_n_dict[i] = neuron_n
                else:
                    neuron_n = neuron_n_dict[i]
                level = self.group_level_dict[neuron_name]
                if level not in self.level_neuron_ct.keys():
                    self.level_neuron_ct[level] = neuron_n
                else:
                    self.level_neuron_ct[level] += neuron_n
                total_in = int(len(child_feat) + np.sum([self.com_layers[z].out_features for z in child_com])) 
                self.child_dim[neuron_name] = [self.gene_feat]*len(child_feat) + [self.com_layers[z].out_features for z in child_com]
                self.com_layers[neuron_name] = nn.Linear(total_in, neuron_n) 
                self.bn_layers['bn_{}'.format(neuron_name)] = nn.BatchNorm1d(neuron_n)
                if self.dropout_p > 0:
                    self.dropout_layers['drop_{}'.format(neuron_name)] = nn.Dropout(self.dropout_p)
                self.output_layers['output_{}'.format(neuron_name)] = nn.Linear(neuron_n, self.output_dim)
                neuron_to_build.remove(i)
                self.build_order.append(i)
        print("Successfully build layers!")
        return neuron_n_dict

    def report_parameter_n(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info("Total {} parameters and {} are trainable".format(total_params, trainable_params))
        return trainable_params

    def forward(self, features): 
        if self.for_lr_finder:
            features = [features[:, i].reshape(features.shape[0], -1) for i in range(features.shape[1])] 
        features = features + self.output  

        pred = [None] * len(self.build_order) 
        states = [None] * len(self.build_order) 
        
        for i in self.build_order:
            j = i + self.input_dim
            neuron_name = self.idx_name[j]
            
            com_layer = self.com_layers[neuron_name]
            bn_layer = self.bn_layers['bn_{}'.format(neuron_name)]
            
            children = self.child_map[i]
            if neuron_name in self.only_combine_gene_group_dict:
                children = [z for z in children if z >= self.input_dim]
            input_list = [features[z] for z in children]
            input_mat = torch.cat(input_list, axis=1)
            features[j] = com_layer(input_mat)
            state = self.act_func(features[j])
            states[i] = state
            features[j] = bn_layer(state)

            if self.dropout_p > 0:
                drop_layer = self.dropout_layers['drop_{}'.format(neuron_name)]
                features[j] = drop_layer(features[j])
            output_layer = self.output_layers['output_{}'.format(neuron_name)]
            if self.use_sigmoid_output:
                pred[i] = self.sigmoid(output_layer(features[j]))
            else:
                pred[i] = output_layer(features[j])

        out = pred[-1]
        return out

class MLP_drug(torch.nn.Module):

    def __init__(
        self,
        sizes,
        batch_norm=True,
        last_layer_act="linear",
        append_layer_width=None,
        append_layer_position=None,
    ):
        super(MLP_drug, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 2
                else None,
                torch.nn.ReLU(),
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        if self.activation == "linear":
            pass
        elif self.activation == "ReLU":
            self.relu = torch.nn.ReLU()
        else:
            raise ValueError("last_layer_act must be one of 'linear' or 'ReLU'")

        if append_layer_width:
            assert append_layer_position in ("first", "last")
            if append_layer_position == "first":
                layers_dict = OrderedDict()
                layers_dict["append_linear"] = torch.nn.Linear(
                    append_layer_width, sizes[0]
                )
                layers_dict["append_bn1d"] = torch.nn.BatchNorm1d(sizes[0])
                layers_dict["append_relu"] = torch.nn.ReLU()
                for i, module in enumerate(layers):
                    layers_dict[str(i)] = module
            else:
                layers_dict = OrderedDict(
                    {str(i): module for i, module in enumerate(layers)}
                )
                layers_dict["append_bn1d"] = torch.nn.BatchNorm1d(sizes[-1])
                layers_dict["append_relu"] = torch.nn.ReLU()
                layers_dict["append_linear"] = torch.nn.Linear(
                    sizes[-1], append_layer_width
                )
        else:
            layers_dict = OrderedDict(
                {str(i): module for i, module in enumerate(layers)}
            )

        self.network = torch.nn.Sequential(layers_dict)

    def forward(self, x):
        if self.activation == "ReLU":
            x = self.network(x)
            dim = x.size(1) // 2
            return torch.cat((self.relu(x[:, :dim]), x[:, dim:]), dim=1)
        return self.network(x)
    
class MLP(torch.nn.Module):

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        return self.network(x)


class GEARS_Model(torch.nn.Module):

    def __init__(self, args):

        super(GEARS_Model, self).__init__()
        self.args = args   
        self.gene_list = args['gene_list']    
        self.num_genes = args['num_genes']
        # self.num_perts = args['num_perts']
        hidden_size = args['hidden_size']
        self.uncertainty = args['uncertainty']
        self.num_layers = args['num_go_gnn_layers']
        self.indv_out_hidden_size = args['decoder_hidden_size']
        self.num_layers_gene_pos = args['num_gene_gnn_layers']
        self.no_perturb = args['no_perturb']
        self.pert_emb_lambda = 0.2
        self.canon_smiles_unique_sorted = args['smiles']
        self.dosages_list = args['dosages']

        with open("./biovnn/BioVNN_pre.pkl","rb") as f:
            self.biovnn_dict=pickle.load(f)
        self.only_combine_child_gene_group = False
        self.neuron_ratio = 1
        

        self.hparams = {
            'dosers_width': 64,
            'dosers_depth': 3,
            'embedding_encoder_width' : 128,
            'embedding_encoder_depth' : 4,
            'dim' : 64 
        }
        self.drug_embeddings = get_chemical_representation(
            smiles=self.canon_smiles_unique_sorted,
            embedding_model='rdkit',
            device=self.args['device'],
        )

        self.dosers = MLP_drug(
            [self.drug_embeddings.embedding_dim + 1]
            + [self.hparams["dosers_width"]] * self.hparams["dosers_depth"]
            + [1],
        )

        self.drug_embedding_encoder = MLP_drug(
            [self.drug_embeddings.embedding_dim]
            + [self.hparams["embedding_encoder_width"]]
            * self.hparams["embedding_encoder_depth"]
            + [self.hparams["dim"]],
            last_layer_act="linear",
        )
        self.pert_w = nn.Linear(1, hidden_size)
          
        self.gene_emb = nn.Embedding(self.num_genes, hidden_size, max_norm=True)

        self.emb_trans = nn.ReLU()
        self.pert_base_trans = nn.ReLU()
        self.transform = nn.ReLU()
        self.emb_trans_v2 = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        self.pert_fuse = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')

        self.G_coexpress = args['G_coexpress'].to(args['device'])
        self.G_coexpress_weight = args['G_coexpress_weight'].to(args['device'])

        self.emb_pos = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.layers_emb_pos = torch.nn.ModuleList()
        for i in range(1, self.num_layers_gene_pos + 1):
            self.layers_emb_pos.append(SGConv(hidden_size, hidden_size, 1))

        self.sim_layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            self.sim_layers.append(SGConv(hidden_size, hidden_size, 1))

        self.recovery_w = MLP([hidden_size, hidden_size*2, hidden_size], last_layer_act='linear')

        self.indv_w1 = nn.Parameter(torch.rand(self.num_genes,
                                               hidden_size, 1))
        self.indv_b1 = nn.Parameter(torch.rand(self.num_genes, 1))
        self.act = nn.ReLU()
        nn.init.xavier_normal_(self.indv_w1)
        nn.init.xavier_normal_(self.indv_b1)

        self.cross_gene_state = MLP([self.num_genes, hidden_size,
                                     hidden_size])
        self.indv_w2 = nn.Parameter(torch.rand(1, self.num_genes,
                                           hidden_size+1))
        self.indv_b2 = nn.Parameter(torch.rand(1, self.num_genes))
        nn.init.xavier_normal_(self.indv_w2)
        nn.init.xavier_normal_(self.indv_b2)

        self.bn_emb = nn.BatchNorm1d(hidden_size)

        self.bn_pert_base = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base_trans = nn.BatchNorm1d(hidden_size)

        self.VNN_cell = VNN_cell(input_dim=self.num_genes, 
                                    output_dim=hidden_size,
                                    biovnn_dict=self.biovnn_dict,
                                    only_combine_child_gene_group=self.only_combine_child_gene_group,
                                    dropout_p=0.1) 
        
        if self.uncertainty:
            self.uncertainty_w = MLP([hidden_size, hidden_size*2, hidden_size, 1], last_layer_act='linear')

    def compute_drug_embeddings_(self, drugs=None, drugs_idx=None, dosages=None):
        """
        Compute sum of drug embeddings, each of them multiplied by its
        dose-response curve.

        If use_drugs_idx is True, then drugs_idx and dosages will be set.
        If use_drugs_idx is False, then drugs will be set.

        @param drugs: A vector of dim [batch_size, num_drugs], where each entry contains the dose of that drug.
        @param drugs_idx: A vector of dim [batch_size]. Each entry contains the index of the applied drug. The
            index is ∈ [0, num_drugs).
        @param dosages: A vector of dim [batch_size]. Each entry contains the dose of the applied drug.
        @return: a tensor of shape [batch_size, drug_embedding_dimension]
        """
        assert (drugs is not None) or (drugs_idx is not None and dosages is not None)
        latent_drugs = self.drug_embeddings.weight

        if drugs is None:
            if len(drugs_idx.size()) == 0:
                drugs_idx = drugs_idx.unsqueeze(0)

            if len(dosages.size()) == 0:
                dosages = dosages.unsqueeze(0)

        if drugs_idx is not None:
            assert drugs_idx.shape == dosages.shape and len(drugs_idx.shape) == 1
            latent_drugs = latent_drugs[drugs_idx]

        scaled_dosages = self.dosers(
            torch.concat([latent_drugs, torch.unsqueeze(dosages, dim=-1)], dim=1)
        ).squeeze()

        if len(scaled_dosages.size()) == 0:
            scaled_dosages = scaled_dosages.unsqueeze(0)

        latent_drugs = self.drug_embedding_encoder(latent_drugs)

        if drugs_idx is None:
            return scaled_dosages @ latent_drugs
        else:

            return torch.einsum("b,be->be", [scaled_dosages, latent_drugs])

    def forward(self, data):
        """
        Forward pass of the model
        """
        x, drugs_idx, dosages, drugs = data.x, data.pert_idx, data.dose, data.pert
        if self.no_perturb:
            out = x.reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)           
            return torch.stack(out)
        else:
            num_graphs = len(data.batch.unique())

            emb = self.gene_emb(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))   

            emb = self.bn_emb(emb)
            base_emb = self.emb_trans(emb)  

            pos_emb = self.emb_pos(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))
            for idx, layer in enumerate(self.layers_emb_pos):
                pos_emb = layer(pos_emb, self.G_coexpress, self.G_coexpress_weight)
                if idx < len(self.layers_emb_pos) - 1:
                    pos_emb = pos_emb.relu()

            base_emb = base_emb + 0.2 * pos_emb
            base_emb = self.emb_trans_v2(base_emb)

            assert (drugs is not None) or (drugs_idx is not None and dosages is not None)
            drugs_idx = torch.tensor(drugs_idx).reshape(len(drugs_idx)).to(self.args['device'])
            dosages = torch.tensor(dosages, dtype=torch.float32).to(self.args['device'])
            drugs = np.array(drugs)
            assert (drugs is not None) or (drugs_idx is not None and dosages is not None)

            latent_drugs = self.drug_embeddings.weight

            if drugs is None:
                if len(drugs_idx.size()) == 0:
                    drugs_idx = drugs_idx.unsqueeze(0)

                if len(dosages.size()) == 0:
                    dosages = dosages.unsqueeze(0)

            if drugs_idx is not None:
                assert drugs_idx.shape == dosages.shape and len(drugs_idx.shape) == 1
                latent_drugs = latent_drugs[drugs_idx]

            scaled_dosages = self.dosers(
                torch.concat([latent_drugs, torch.unsqueeze(dosages, dim=-1)], dim=1)
            ).squeeze()

            if len(scaled_dosages.size()) == 0:
                scaled_dosages = scaled_dosages.unsqueeze(0)

            latent_drugs = self.drug_embedding_encoder(latent_drugs)

            if drugs_idx is None:
                drug_embedding = scaled_dosages @ latent_drugs
            else:
                drug_embedding = torch.einsum("b,be->be", [scaled_dosages, latent_drugs])

            base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)
            drug_embedding = drug_embedding.unsqueeze(1).repeat(1, self.num_genes, 1)

            base_emb = base_emb + drug_embedding
            base_emb = base_emb.reshape(drugs_idx.shape[0] * self.num_genes, -1)
            base_emb = self.bn_pert_base(base_emb)
        
            base_emb = self.transform(base_emb)        
            out = self.recovery_w(base_emb)
            out = out.reshape(drugs_idx.shape[0], self.num_genes, -1)
            out = out.unsqueeze(-1) * self.indv_w1
            w = torch.sum(out, axis = 2)
            out = w + self.indv_b1

            cross_gene_embed = self.VNN_cell(out.reshape(drugs_idx.shape[0], self.num_genes, -1).squeeze(2))
            cross_gene_embed = cross_gene_embed.repeat(1, self.num_genes)

            cross_gene_embed = cross_gene_embed.reshape([drugs_idx.shape[0],self.num_genes, -1])
            cross_gene_out = torch.cat([out, cross_gene_embed], 2)

            cross_gene_out = cross_gene_out * self.indv_w2
            cross_gene_out = torch.sum(cross_gene_out, axis=2)
            out = cross_gene_out + self.indv_b2        

            out = out.reshape(drugs_idx.shape[0] * self.num_genes, -1) + x.reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)

            ## uncertainty head
            if self.uncertainty:
                out_logvar = self.uncertainty_w(base_emb)
                out_logvar = torch.split(torch.flatten(out_logvar), self.num_genes)
                return torch.stack(out), torch.stack(out_logvar)
            
            return torch.stack(out)