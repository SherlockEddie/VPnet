import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchmetrics import R2Score
from sklearn.metrics import r2_score
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from scipy.stats import pearsonr
from torch_geometric.data import Batch
from sklearn.metrics import mean_squared_error as mse

def repeat_n(x, n):
    return x.view(1, -1).repeat(n, 1)

def bool2idx(x):

    return np.where(x)[0]

def evaluate_r2(model, dataset, control_dataset, device):
    model.eval()
    model = model.to(device)
    mean_score, pcc_score, mean_score_de, pcc_score_de, mse_score, mse_score_de = [], [], [], [], [], []
    genes_control = control_dataset.genes

    celltypes = np.unique(control_dataset.covariate_names['cell_type'])
    celltype_index  = {}
    for i in celltypes:
        celltype_index[i] = bool2idx(control_dataset.covariate_names['cell_type'] == i)
    pert_categories_index = pd.Index(dataset.pert_categories, dtype="category")
    for cell_drug_dose_comb, category_count in tqdm(zip(
        *np.unique(dataset.pert_categories, return_counts=True)
    ), total=len(np.unique(dataset.pert_categories)), desc="Evaluate Testdataset"):
        drug_info = cell_drug_dose_comb.split('_')
        type_sel = drug_info[0]
        drugname_sel = drug_info[1]

        genes_test = genes_control[celltype_index[type_sel]]
        
        n_rows = genes_test.size(0)

        if dataset.perturbation_key is None:
            break

        if category_count <= 5:
            continue

        if (
            "dmso" in cell_drug_dose_comb.lower()
            or "control" in cell_drug_dose_comb.lower()
        ):
            continue

        bool_de = dataset.var_names.isin(
            np.array(dataset.de_genes['_'.join(drug_info[:-1])])
        )
        idx_de = bool2idx(bool_de)

        if len(idx_de) < 2:
            continue

        bool_category = pert_categories_index.get_loc(cell_drug_dose_comb)
        idx_all = bool2idx(bool_category)
        idx = idx_all[0]

        cell_graphs = []
        dose = dataset.dosages[idx]
        drug_idx = [dataset.drugs_idx[idx].item()]
        for x in genes_test:
            cell_graphs.append(Data(x=torch.Tensor(np.array(x).reshape(1, -1)).T, pert_idx=drug_idx,
                de_idx=idx_de, dose=dose, pert=drugname_sel))
        test_loader = DataLoader(cell_graphs, batch_size=512, shuffle=False, drop_last = False)
        yp_all = torch.tensor([]).to(device)
        for step, batch in enumerate(test_loader):
            batch = batch.to(device)
            with torch.no_grad():
                y_pred = model(batch)
            yp_all = torch.cat((yp_all, y_pred), dim = 0)

        y_true = dataset.genes[idx_all, :].to(device)
        yt_m = y_true.mean(dim=0)
        yp_m = yp_all.mean(dim=0)


        r2_m = compute_r2(yt_m, yp_m)
        pcc_m = pearsonr(yt_m.cpu().numpy(), yp_m.cpu().numpy())[0]
        mse_m = mse(yt_m.cpu().numpy(), yp_m.cpu().numpy())
        r2_m_de = compute_r2(yt_m[idx_de], yp_m[idx_de])
        pcc_m_de = pearsonr(yt_m[idx_de].cpu().numpy(), yp_m[idx_de].cpu().numpy())[0]
        mse_m_de = mse(yt_m[idx_de].cpu().numpy(), yp_m[idx_de].cpu().numpy())

        if r2_m_de == float("-inf"): #or r2_v_de == float("-inf"):
            continue

        mean_score.append(max(r2_m, 0.0))
        pcc_score.append(max(pcc_m, 0.0))
        mse_score.append(max(mse_m, 0.0))
        mean_score_de.append(max(r2_m_de, 0.0))
        pcc_score_de.append(max(pcc_m_de, 0.0))
        mse_score_de.append(max(mse_m_de, 0.0))

    if len(mean_score) > 0:
        return {
            'r2' : np.mean(mean_score),
            'r2_de' : np.mean(mean_score_de),
            'pearson' : np.mean(pcc_score),
            'pearson_de' : np.mean(pcc_score_de),
            'mse' : np.mean(mse_score),
            'mse_de' : np.mean(mse_score_de),
        }
    else:
        return []

def evaluate_test(loader, model, uncertainty, dose_tofind, device):

    model.eval()
    model.to(device)
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}
    logvar = []
    
    for itr, batch in enumerate(loader):
        dosages = torch.tensor([dose[0] for dose in batch.dose], dtype=torch.float32)
        indices = np.array(torch.where(dosages == dose_tofind)[0])
        batch.to(device)
        pert = batch.pert
        perts = [pert[i] for i in indices]
        pert_cat.extend(perts)

        with torch.no_grad():
            if uncertainty:
                p, unc = model(batch)
                logvar.extend(unc.cpu())
            else:
                p = model(batch)
            t = batch.y
            pred.extend(p[indices].cpu())
            truth.extend(t[indices].cpu())
            
            # Differentially expressed genes
            for itr, de_idx in enumerate(np.array(batch.de_idx)[indices].tolist()):
                pred_de.append(p[itr, de_idx])
                truth_de.append(t[itr, de_idx])

    # all genes
    results['pert_cat'] = np.array(pert_cat)
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results['pred']= pred.detach().cpu().numpy()
    results['truth']= truth.detach().cpu().numpy()

    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)
    results['pred_de']= pred_de.detach().cpu().numpy()
    results['truth_de']= truth_de.detach().cpu().numpy()
    
    if uncertainty:
        results['logvar'] = torch.stack(logvar).detach().cpu().numpy()
    
    return results

def evaluate(loader, model, uncertainty, device):
    model = model.to(device)
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}
    logvar = []
    
    for itr, batch in tqdm(enumerate(loader),  total=len(loader), desc="Evaluate"):
        batch = batch.to(device)
        pert_cat.extend(batch.pert)

        with torch.no_grad(): 
            if uncertainty:
                p, unc = model(batch)
                logvar.extend(unc.cpu())
            else:
                p = model(batch)
            t = batch.y
            pred.extend(p.cpu())
            truth.extend(t.cpu())
            
            # Differentially expressed genes
            for itrs, de_idx in enumerate(batch.de_idx):
                pred_de.append(p[itrs, de_idx])
                truth_de.append(t[itrs, de_idx])

    # all genes
    results['pert_cat'] = np.array(pert_cat)
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results['pred']= pred.detach().cpu().numpy()
    results['truth']= truth.detach().cpu().numpy()

    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)
    results['pred_de']= pred_de.detach().cpu().numpy()
    results['truth_de']= truth_de.detach().cpu().numpy()

    # print(results['pred_de'])
    
    if uncertainty:
        results['logvar'] = torch.stack(logvar).detach().cpu().numpy()
    
    return results

def compute_r2(y_true, y_pred):
    y_pred = torch.clamp(y_pred, -3e12, 3e12)
    metric = R2Score().to(y_true.device)
    metric.update(y_pred, y_true)  # same as sklearn.r2_score(y_true, y_pred)
    return metric.compute().item()


def compute_metrics(results):
    metrics = {}
    metrics_pert = {}
    draw_de = {}

    metric2fct = {
           'mse': mse,
           'pearson': pearsonr,
           'r2' : r2_score
    }
    
    for m in metric2fct.keys():
        metrics[m] = []
        metrics[m + '_de'] = []

    for pert in np.unique(results['pert_cat']):
        if pert == 'control':
            continue

        metrics_pert[pert] = {}
        p_idx = np.where(results['pert_cat'] == pert)[0]
            
        for m, fct in metric2fct.items():
            if m == 'pearson':
                val = fct(results['pred'][p_idx].mean(0), results['truth'][p_idx].mean(0))[0]
                
                if np.isnan(val):
                    val = 0
            elif m == 'mse':
                val = fct(results['truth'][p_idx].mean(0), results['pred'][p_idx].mean(0))
            else:
                val = fct(results['truth'][p_idx].mean(0), results['pred'][p_idx].mean(0))
                if np.isnan(val):
                    val = 0

            metrics_pert[pert][m] = val
            metrics[m].append(metrics_pert[pert][m])

       
        if pert != 'control':
            
            for m, fct in metric2fct.items():
                if m == 'pearson':
                    val = fct(results['pred_de'][p_idx].mean(0), results['truth_de'][p_idx].mean(0))[0]
                    if np.isnan(val):
                        val = 0
                elif m == 'mse':
                    val = fct(results['truth_de'][p_idx].mean(0), results['pred_de'][p_idx].mean(0))
                else:
                    val = fct(results['truth_de'][p_idx].mean(0), results['pred_de'][p_idx].mean(0))
                    if np.isnan(val):
                        val = 0
                    
                metrics_pert[pert][m + '_de'] = val
                metrics[m + '_de'].append(metrics_pert[pert][m + '_de'])

                draw_de[pert + 'truth'] = results['truth_de'][p_idx].mean(0)
                draw_de[pert + 'pred'] = results['pred_de'][p_idx].mean(0)

        else:
            for m, fct in metric2fct.items():
                metrics_pert[pert][m + '_de'] = 0
    
    for m in metric2fct.keys():
        
        metrics[m] = np.mean(metrics[m])
        metrics[m + '_de'] = np.mean(metrics[m + '_de'])

    return metrics, metrics_pert




