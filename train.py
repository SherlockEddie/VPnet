from vpnet.data import init_dataset, load_pyg
from threadpoolctl import threadpool_limits
from vpnet.gearscp import GEARS
import torch

with threadpool_limits(user_api='openmp', limits=4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_config = {
        'perturbation_key': 'condition', 
        'pert_category': 'cov_drug_dose_name', 
        'dose_key': 'dose', 
        'covariate_keys': 'cell_type', 
        'smiles_key': 'SMILES', 
        'use_drugs_idx': True, 
        'split_key': 'split', 
        'dataset_path': 'data/sciplex_lincs.h5ad', 
        'degs_key': 'rank_genes_groups_cov'
    }

    datasets, dataset = init_dataset(dataset_config)
    pyg_path = "./data/data_pyg"
    dataset = load_pyg(dataset, pyg_path)

    dataset.get_dataloader(batch_size = 512)
    gears_model = GEARS(dataset, datasets, device = device)
    gears_model.model_initialize(hidden_size = 64)
    # gears_model.load_pretrained('./model')
    gears_model.train() 