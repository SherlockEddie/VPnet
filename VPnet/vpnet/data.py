import logging
import warnings
from typing import List, Optional, Union
import os
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import biovnn.utils_biovnn as ub
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data.storage import GlobalStorage
from .utils import print_sys

from rdkit import Chem

warnings.simplefilter(action="ignore", category=FutureWarning)

    
def canonicalize_smiles(smiles: Optional[str]):
    if smiles:
        return Chem.CanonSmiles(smiles)
    else:
        return None
    
def ranks_to_df(data, key="rank_genes_groups"):

    d = data.uns[key]
    dfs = []
    for k in d.keys():
        if k == "params":
            continue
        series = pd.DataFrame.from_records(d[k]).unstack()
        series.name = k
        dfs.append(series)

    return pd.concat(dfs, axis=1)


def drug_names_to_once_canon_smiles(
    drug_names: List[str], dataset: sc.AnnData, perturbation_key: str, smiles_key: str
):

    name_to_smiles_map = {
        # drug: canonicalize_smiles(smiles)
        drug: smiles
        for drug, smiles in dataset.obs.groupby(
            [perturbation_key, smiles_key]
        ).groups.keys()
    }
    return [name_to_smiles_map[name] for name in drug_names]


indx = lambda a, i: a[i] if a is not None else None


class Dataset:
    covariate_keys: Optional[List[str]]
    drugs: torch.Tensor  # stores the (OneHot * dosage) encoding of drugs / combinations of drugs
    drugs_idx: torch.Tensor  # stores the integer index of the drugs applied to each cell.
    max_num_perturbations: int  # how many drugs are applied to each cell at the same time?
    dosages: torch.Tensor  # shape: (dataset_size, max_num_perturbations)
    drugs_names_unique_sorted: np.ndarray  # sorted list of all drug names in the dataset

    def __init__(
        self,
        data: str,
        perturbation_key=None,
        dose_key=None,
        covariate_keys=None,
        smiles_key=None,
        degs_key="rank_genes_groups_cov",
        pert_category="cov_drug_dose_name",
        split_key="split",
        use_drugs_idx=False,
    ):
        print_sys(f"Starting to read in data: {data}\n...")
        if isinstance(data, AnnData):
            data = data
            # sc.pp.normalize_total(data)
            # sc.pp.log1p(data)
        else:
            data = sc.read(data)
            # sc.pp.normalize_total(data)
            # sc.pp.log1p(data)
        print_sys(f"Finished data loading.")
        self.adata = data
        self.genes = torch.Tensor(data.X)
        self.var_names = data.var_names

        '''
        Preprocess BioVNN data
        '''
        if not os.path.exists("./biovnn/BioVNN_pre.pkl"):
            biovnn_pre=ub.BioVNN_pre(data.obs_names, list(self.var_names)) 
            biovnn_pre.perform() 

        self.perturbation_key = perturbation_key
        self.dose_key = dose_key
        if isinstance(covariate_keys, str):
            covariate_keys = [covariate_keys]
        self.covariate_keys = covariate_keys
        self.smiles_key = smiles_key
        self.use_drugs_idx = use_drugs_idx
        self.train_processed = None
        self.val_processed = None
        self.test_processed = None
        self.set2conditions = None
        self.seed = 1

        if perturbation_key is not None:
            if dose_key is None:
                raise ValueError(
                    f"A 'dose_key' is required when provided a 'perturbation_key'({perturbation_key})."
                )
            self.pert_categories = np.array(data.obs[pert_category].values)
            self.de_genes = data.uns[degs_key]
            self.drugs_names = np.array(data.obs[perturbation_key].values)
            self.dose_names = np.array(data.obs[dose_key].values)

            drugs_names_unique = set(self.drugs_names)

            self.drugs_names_unique_sorted = np.array(sorted(drugs_names_unique))

            self._drugs_name_to_idx = {
                smiles: idx for idx, smiles in enumerate(self.drugs_names_unique_sorted)
            }
            self.canon_smiles_unique_sorted = drug_names_to_once_canon_smiles(
                list(self.drugs_names_unique_sorted), data, perturbation_key, smiles_key
            )
            self.max_num_perturbations = max(
                len(name.split("*")) for name in self.drugs_names
            )

            if not use_drugs_idx:
                self.encoder_drug = OneHotEncoder(
                    sparse=False, categories=[list(self.drugs_names_unique_sorted)]
                )
                self.encoder_drug.fit(self.drugs_names_unique_sorted.reshape(-1, 1))
                self.atomic_drugs_dict = dict(
                    zip(
                        self.drugs_names_unique_sorted,
                        self.encoder_drug.transform(
                            self.drugs_names_unique_sorted.reshape(-1, 1)
                        ),
                    )
                )
                drugs = []
                for i, comb in enumerate(self.drugs_names):
                    drugs_combos = self.encoder_drug.transform(
                        np.array(comb).reshape(-1, 1)
                    )
                    dose_combos = str(data.obs[dose_key].values[i]).split("+")
                    for j, d in enumerate(dose_combos):
                        if j == 0:
                            drug_ohe = float(d) * drugs_combos[j]
                        else:
                            drug_ohe += float(d) * drugs_combos[j]
                    drugs.append(drug_ohe)
                self.drugs = torch.Tensor(np.array(drugs))

                self.drug_dict = {}
                atomic_ohe = self.encoder_drug.transform(
                    self.drugs_names_unique_sorted.reshape(-1, 1)
                )
                for idrug, drug in enumerate(self.drugs_names_unique_sorted):
                    i = np.where(atomic_ohe[idrug] == 1)[0][0]
                    self.drug_dict[i] = drug
            else:
                assert (
                    self.max_num_perturbations == 1
                ), "Index-based drug encoding only works with single perturbations"
                drugs_idx = [self.drug_name_to_idx(drug) for drug in self.drugs_names]
                self.drugs_idx = torch.tensor(
                    drugs_idx,
                    dtype=torch.long,
                )
                dosages = [float(dosage) for dosage in self.dose_names]
                self.dosages = torch.tensor(
                    dosages,
                    dtype=torch.float32,
                )

        else:
            self.pert_categories = None
            self.de_genes = None
            self.drugs_names = None
            self.dose_names = None
            self.drugs_names_unique_sorted = None
            self.atomic_drugs_dict = None
            self.drug_dict = None
            self.drugs = None

        if isinstance(covariate_keys, list) and covariate_keys:
            if not len(covariate_keys) == len(set(covariate_keys)):
                raise ValueError(f"Duplicate keys were given in: {covariate_keys}")
            self.covariate_names = {}
            self.covariate_names_unique = {}
            self.atomic_сovars_dict = {}
            self.covariates = []
            for cov in covariate_keys:
                self.covariate_names[cov] = np.array(data.obs[cov].values)
                self.covariate_names_unique[cov] = np.unique(self.covariate_names[cov])

                names = self.covariate_names_unique[cov]
                encoder_cov = OneHotEncoder(sparse=False)
                encoder_cov.fit(names.reshape(-1, 1))

                self.atomic_сovars_dict[cov] = dict(
                    zip(list(names), encoder_cov.transform(names.reshape(-1, 1)))
                )

                names = self.covariate_names[cov]
                self.covariates.append(
                    torch.Tensor(encoder_cov.transform(names.reshape(-1, 1))).float()
                )
        else:
            self.covariate_names = None
            self.covariate_names_unique = None
            self.atomic_сovars_dict = None
            self.covariates = None

        self.ctrl = data.obs["control"].values

        if perturbation_key is not None:
            self.ctrl_name = list(
                np.unique(data[data.obs["control"] == 1].obs[self.perturbation_key])
            )
        else:
            self.ctrl_name = None

        if self.covariates is not None:
            self.num_covariates = [
                len(names) for names in self.covariate_names_unique.values()
            ]
        else:
            self.num_covariates = [0]
        self.num_genes = self.genes.shape[1]
        self.num_drugs = (
            len(self.drugs_names_unique_sorted)
            if self.drugs_names_unique_sorted is not None
            else 0
        )

        test_data = data[data.obs[split_key] == "test"]
        test_data = test_data[test_data.obs['dose'].astype(float)!=0.0]
        obsname = data.obs_names
        self.indices = {
            "all": list(range(len(self.genes))),
            "control": np.where(data.obs['dose'].astype(float)==0.0)[0].tolist(),
            "treated": np.where(data.obs['dose'].astype(float)!=0.0)[0].tolist(),
            "test": np.where(data.obs[split_key] == "test")[0].tolist(),
            "test_paired": obsname.get_indexer(test_data.obs['paired_control_index'].values),
        }

        # This could be made much faster
        degs_tensor = []
        # check if the DEGs are condtioned on covariate, drug, and dose or only covariate and drug
        dose_specific = (
            True if len(list(self.de_genes.keys())[0].split('_')) == 3 else False
        )
        for i in range(len(self)):
            drug = indx(self.drugs_names, i)
            cov = indx(self.covariate_names["cell_type"], i)
            if dose_specific:
                dose = indx(self.dose_names, i)

            if drug == "JQ1":
                drug = "(+)-JQ1"

            if drug == "control" or drug == "DMSO":
                genes = []
            else:
                key = f"{cov}_{drug}_{dose}" if dose_specific else f"{cov}_{drug}"
                genes = self.de_genes[key]
            degs_tensor.append(
                torch.Tensor(self.var_names.isin(genes)).detach().clone()
            )
        self.degs = torch.stack(degs_tensor)

    def subset(self, split, condition="all"):
        idx = list(set(self.indices[split]) & set(self.indices[condition]))
        return SubDataset(self, idx)

    def drug_name_to_idx(self, drug_name: str):
        """
        For the given drug, return it's index. The index will be persistent for each dataset (since the list is sorted).
        Raises ValueError if the drug doesn't exist in the dataset.
        """
        return self._drugs_name_to_idx[drug_name]
    
    def get_dataloader(self, batch_size):
            
        self.node_map = {x: it for it, x in enumerate(self.adata.var_names)}
        self.gene_names = self.adata.var_names
       
        # Create cell graphs
        cell_graphs = {}
        cell_graphs['train'] = []
        for p in self.set2conditions['train']:
            if p == 'control':
                continue
            cell_graphs['train'].extend(self.train_processed[p])

        cell_graphs['val'] = []
        for p in self.set2conditions['val']:
            if p == 'control':
                continue
            cell_graphs['val'].extend(self.val_processed[p])

        cell_graphs['test'] = []
        for p in self.set2conditions['test']:
            if p == 'control':
                continue
            cell_graphs['test'].extend(self.test_processed[p])
        
        # Set up dataloaders
        train_loader = DataLoader(cell_graphs['train'],
                            batch_size=batch_size, shuffle=True, drop_last = True)
        val_loader = DataLoader(cell_graphs['val'],
                            batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(cell_graphs['test'],
                            batch_size=batch_size, shuffle=False)
        
        self.dataloader =  {'train_loader': train_loader,
                                'val_loader': val_loader,
                                'test_loader': test_loader,}
        print_sys("Done!")

    def __getitem__(self, i):
        if self.use_drugs_idx:
            return (
                self.genes[i],
                indx(self.drugs_idx, i),
                indx(self.dosages, i),
                indx(self.degs, i),
                *[indx(cov, i) for cov in self.covariates],
            )
        else:
            return (
                self.genes[i],
                indx(self.drugs, i),
                indx(self.degs, i),
                *[indx(cov, i) for cov in self.covariates],
            )

    def __len__(self):
        return len(self.genes)


class SubDataset:
    """
    Subsets a `Dataset` by selecting the examples given by `indices`.
    """

    def __init__(self, dataset: Dataset, indices):
        self.perturbation_key = dataset.perturbation_key
        self.adata = dataset.adata[indices]
        self.dose_key = dataset.dose_key
        self.covariate_keys = dataset.covariate_keys
        self.smiles_key = dataset.smiles_key
        self.dose = dataset.dose_names[indices]
        self.drugs_names_unique_sorted = dataset.drugs_names_unique_sorted
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'control']
        self.obs_names = self.adata.obs_names


        self.covars_dict = dataset.atomic_сovars_dict

        self.genes = dataset.genes[indices]
        self.use_drugs_idx = dataset.use_drugs_idx
        if self.use_drugs_idx:
            self.drugs_idx = indx(dataset.drugs_idx, indices)
            self.dosages = indx(dataset.dosages, indices)
        else:
            self.perts_dict = dataset.atomic_drugs_dict
            self.drugs = indx(dataset.drugs, indices)
        self.covariates = [indx(cov, indices) for cov in dataset.covariates]

        self.drugs_names = indx(dataset.drugs_names, indices)
        self.pert_categories = indx(dataset.pert_categories, indices)
        self.covariate_names = {}
        assert (
            "cell_type" in self.covars_dict
        ), "`cell_type` must be provided as a covariate"
        for cov in self.covariate_keys:
            self.covariate_names[cov] = indx(dataset.covariate_names[cov], indices)

        self.var_names = dataset.var_names
        self.de_genes = dataset.de_genes
        self.ctrl_name = indx(dataset.ctrl_name, 0)

        self.num_covariates = dataset.num_covariates
        self.num_genes = dataset.num_genes
        self.num_drugs = dataset.num_drugs

        self.degs = dataset.degs

    def get_pert_idx(self, pert_category):
        """
        Get perturbation index for a given perturbation category

        Parameters
        ----------
        pert_category: str
            Perturbation category

        Returns
        -------
        list
            List of perturbation indices

        """
        try:
            pert_idx = [np.where(p == self.drugs_names_unique_sorted)[0][0]
                    for p in pert_category.split('*')
                    if p != 'control']
        except:
            print(pert_category)
            pert_idx = None
            
        return pert_idx

    def create_cell_graph(self, X, y, de_idx, pert, dose, pert_idx=None):
        feature_mat = torch.Tensor(X).T
        if pert_idx is None:
            pert_idx = [-1]
        return Data(x=feature_mat, pert_idx=pert_idx,
                    y=torch.Tensor(y), de_idx=de_idx, dose=dose, pert=pert)

    def create_cell_graph_dataset(self, split_adata, pert_category,
                                  num_samples=1):

        num_de_genes = 50        
        adata_ = split_adata[split_adata.obs['condition'] == pert_category]
        Xs = []
        ys = []
        dose_all = []
        de_index_all = []
        cell_graphs = []
        # When considering a non-control perturbation
        if pert_category != 'control':
            # Get the indices of applied perturbation
            pert_idx = self.get_pert_idx(pert_category)

            # Store list of genes that are most differentially expressed for testing
            pert_de_category = adata_.obs['condition'][0]

            obsname = adata_[0].obs_names[0]
            degene = self.degs[self.obs_names.get_loc(obsname)]
            indices = (torch.where(degene == 1)[0]).numpy()

            for i in adata_.obs['cell_type'].unique():
                sub_adata_ = adata_[adata_.obs['cell_type'] == i] # N*977
                ctrl = self.ctrl_dict[i] # M*977
                sample_index = np.random.randint(0, len(ctrl), len(sub_adata_))
                ctrl = ctrl.X[sample_index] # N*977
                dose_all.extend(sub_adata_.obs['dose'].values)
                Xs.extend(ctrl.A)
                ys.extend(sub_adata_.X.A)
            
            cell_graphs = []
            for X, y, dose in zip(Xs, ys, dose_all):
                cell_graphs.append(self.create_cell_graph(np.array(X).reshape(1, -1),
                                    np.array(y).reshape(1, -1), indices, pert_category, dose, pert_idx))


        # When considering a control perturbation
        else:
            pert_idx = None
            de_idx = [-1] * num_de_genes
            dose_all.extend(adata_.obs['dose'].values)
            Xs.extend(adata_.X.A)
            ys.extend(adata_.X.A)
            # Create cell graphs
            cell_graphs = []
            for X, y, dose in zip(Xs, ys, dose_all):
                cell_graphs.append(self.create_cell_graph(np.array(X).reshape(1, -1),
                                    np.array(y).reshape(1, -1), de_idx, pert_category, dose, pert_idx))


        return cell_graphs
    
    def create_dataset_file(self, pyg_path):
        """
        Create dataset file for each perturbation condition
        """
        print_sys("Creating dataset file...")
        self.dataset_processed = {}
        for p in tqdm(self.adata.obs['condition'].unique()):
            self.dataset_processed[p] = self.create_cell_graph_dataset(self.adata, p)
        pickle.dump(self.dataset_processed, open(pyg_path, "wb"))
        print_sys("Done!")


    def __getitem__(self, i):
        if self.use_drugs_idx:
            return (
                self.genes[i],
                indx(self.drugs_idx, i),
                indx(self.dosages, i),
                indx(self.degs, i),
                *[indx(cov, i) for cov in self.covariates],
            )
        else:
            return (
                self.genes[i],
                indx(self.drugs, i),
                indx(self.degs, i),
                *[indx(cov, i) for cov in self.covariates],
            )

    def __len__(self):
        return len(self.genes)


def load_dataset_splits(
    dataset_path: str,
    perturbation_key: Union[str, None],
    dose_key: Union[str, None],
    covariate_keys: Union[list, str, None],
    smiles_key: Union[str, None],
    degs_key: str = "rank_genes_groups_cov",
    pert_category: str = "cov_drug_dose_name",
    split_key: str = "split",
    return_dataset: bool = False,
    use_drugs_idx=False,
):
    dataset = Dataset(
        dataset_path,
        perturbation_key,
        dose_key,
        covariate_keys,
        smiles_key,
        degs_key,
        pert_category,
        split_key,
        use_drugs_idx,
    )

    splits = {
        "test": dataset.subset("test", "all"),
        "test_control": dataset.subset("test_paired", "control"),
        "test_treated": dataset.subset("test", "treated"),
    }

    if return_dataset:
        return splits, dataset
    else:
        return splits


def init_dataset(data_params: dict):
    datasets, dataset = load_dataset_splits(
        **data_params, return_dataset=True
    )
    return datasets, dataset


def load_pyg(dataset: Dataset, pyg_path):
    print_sys("Local copy of pyg dataset is detected. Loading...")
    dataset.train_processed = pickle.load(open((pyg_path + "/sciplex_lincs/train_graph.pkl"), "rb"))
    dataset.val_processed = pickle.load(open((pyg_path + "/sciplex_lincs/val_graph.pkl"), "rb"))
    dataset.test_processed = pickle.load(open((pyg_path + "/sciplex_lincs/test_graph.pkl"), "rb"))
    print_sys("Done!")

    map_dict = {
        'train' : 'train',
        'valid' : 'val',
        'test' : 'test'
    }
    dataset.adata.obs['split'] = dataset.adata.obs['split'].map(map_dict)
    set2conditions = dict(dataset.adata.obs.groupby('split').agg({'condition':
                                                lambda x: x}).condition)
    set2conditions = {i: j.unique().tolist() for i,j in set2conditions.items()} 
    dataset.set2conditions = set2conditions

    return dataset