import pandas as pd
import scanpy as sc
import numpy as np
import random
import os
import torch 
from .mitor import HijackingPredictor


def set_random_seed(seed):
    print(f"[MITOR] Setting random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def init_pipeline(adata_path, non_ref_list_path, seed=42):
    set_random_seed(seed)

    print("[MITOR] Loading data...")
    
    adata_RNA = sc.read_h5ad(adata_path)
    
    non_ref_list = pd.read_csv(non_ref_list_path)["mixed_cell_names"]
    
    ref_donor_adata_RNA = adata_RNA[adata_RNA.obs.MTtransfer == 'donor',:]
    ref_receiver_adata_RNA = adata_RNA[(adata_RNA.obs.MTtransfer == 'non-Hijacking')&(~adata_RNA.obs['cell_name'].isin(non_ref_list)),:]
    # ref_receiver_adata_RNA.obs

    ref_adata_dict={}
    ref_adata_dict['MC_T_cell'] = ref_donor_adata_RNA
    ref_adata_dict['MC_cancer_cell'] = ref_receiver_adata_RNA

    predictor_RNA = HijackingPredictor()
    print("[MITOR] Initialization complete")

    return predictor_RNA, ref_adata_dict, non_ref_list ,adata_RNA
    
def run_enMT_profile(predictor_RNA, ref_adata_dict, epoch=80, learning_rate=0.01, weight_decay=1e-3, verbose=False):

    print("[MITOR] Running enMT profile training...")
    predictor_RNA.fit_reference(ref_adata_dict, epoch=epoch, learning_rate=learning_rate, weight_decay=weight_decay, verbose=verbose)
    print("[MITOR] enMT profile training complete.")

def run_predict(predictor_RNA, adata_RNA, non_ref_list, epoch=200, learning_rate=0.1, theta=1e3, verbose=False):
    print("[MITOR] Running final prediction step...")

    mixture_adata = adata_RNA[adata_RNA.obs["cell_name"].isin(non_ref_list), :]

    predictor_RNA.fit_mixture(adata=mixture_adata, epoch=epoch, learning_rate=learning_rate, theta=theta, verbose=verbose)

    predictor_RNA.predict("MC_cancer_cell")
    print("[MITOR] Final prediction complete.")


def get_result(predictor_RNA,mixture_labels,feature='Prob'):
    result = predictor_RNA.get_result(mixture_labels,feature)
    os.makedirs("output", exist_ok=True)
    file_path = os.path.join("output", f'{feature.lower()}_performance_metrics.csv')
    result.to_csv(file_path, index=False)
    print(f"[MITOR] result saved as output/{feature.lower()}_performance_metrics.csv.csv.")


def get_fraction(predictor_RNA):
    df_F = pd.DataFrame(predictor_RNA.F)
    os.makedirs("output", exist_ok=True)
    df_F.to_csv("output/pred_F.csv")
    print("[MITOR] Fraction prediction saved as output/pred_F.csv.")

def plot_result(predictor_RNA, mixture_labels):

    predictor_RNA.eval_pred(mixture_labels)

    print("[MITOR] Plotting complete.")


def plot_info(predictor_RNA, plots=None):
    if plots is None:
        plots = ["ref_profiles", "mixture_cell", "U", "B", "raw_F", "F", "AUROC"]  

    predictor_RNA.plot(plots=plots)

    print("[MITOR] Plotting complete.")