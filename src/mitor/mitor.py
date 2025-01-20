from .model import GammaPoissonModel, GammaPoissonMixModel
import torch
import numpy as np
import pandas as pd
import copy

from sklearn.metrics import auc, brier_score_loss, cohen_kappa_score, confusion_matrix, f1_score, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

import seaborn as sns
from matplotlib import pyplot as plt



class Deconvolutor(object):

    def __init__(self, 
                 device='cpu',
                 prefix='test'):
        
        self.device = device
        self.prefix = prefix
        
        
    def fit_reference(self, ref_adata_dict=None, epoch=80, verbose=False, learning_rate=1e-2,weight_decay = 1e-3,batch_size = 32):
        gp_model_list = []
        self.types = ref_adata_dict.keys()
        for adata in ref_adata_dict.values():
            if type(adata.X) == np.ndarray:
                data = adata.X
            else:
                data = adata.X.toarray()
            gp = GammaPoissonModel(device=self.device,learning_rate=learning_rate,weight_decay=weight_decay)
            
            gp.fit([data], verbose=verbose, epoch=epoch, batch_size=batch_size)
            gp_model_list.append(gp) 
            
        self.gp_model_list = gp_model_list
        self.U = torch.cat([pg.U for pg in gp_model_list]).cpu().detach().numpy()
        self.B = torch.cat([pg.B for pg in gp_model_list]).cpu().detach().numpy()
        self.ref_adata_dict = ref_adata_dict

        
    def fit_mixture(self, adata=None, epoch=10, theta=1, verbose=False,learning_rate=1e-2,batch_size = 32):
        gpm = GammaPoissonMixModel(self.U, self.B, device=self.device,learning_rate=learning_rate)
        data = adata.X
        gpm.fit(data, epoch=epoch, theta=theta, verbose=verbose,batch_size=batch_size)        
        
        self.gpm = gpm
        
        raw_F = torch.nn.functional.relu(gpm.F).detach().cpu().numpy()
         
        row_max = raw_F.max(axis=1, keepdims=True)  
        row_scaled_F = (raw_F) / (row_max +1e-8)

        F = row_scaled_F
        self.raw_F = pd.DataFrame(raw_F.T, columns=self.types, index=adata.obs_names)
        self.F = pd.DataFrame(F.T, columns=self.types, index=adata.obs_names)
        
        adata.uns['raw_F'] = self.raw_F
        adata.uns['F'] = self.F

        self.mixture_cell_adata = adata


class HijackingPredictor(Deconvolutor):

    def __init__(self,
                 device='cpu',
                 prefix='test'):
        
        super().__init__(device=device, prefix=prefix)
         
    def predict(self, donor_type, probs=np.arange(100, -5, -5), ranks=np.arange(100,-5, -5)):
        adata = self._predict_by_probs(donor_type, probs)
        adata = self._predict_by_ranks(donor_type, ranks)
        return adata
        
    def _predict_by_probs(self, donor_type, probs):
        for prob in probs:
            self._predict_by_prob(donor_type, prob)
    
    def _predict_by_prob(self, donor_type, prob):
        adata = self.mixture_cell_adata
        pred = []
        for f in self.F[donor_type]:
            if f <= prob*0.01:
                pred.append('hijacking')
            else:
                pred.append('non-hijacking')
        adata.obs[f'prob{prob}_pred'] = pred    
            
    def _predict_by_ranks(self, donor_type, top_ranks=50):
        for top_rank in top_ranks:
            self._predict_by_rank(donor_type, top_rank=top_rank)
            
    def _predict_by_rank(self, donor_type, top_rank=50):
        adata = self.mixture_cell_adata
        
        N = adata.obs.shape[0]
        f = self.F[donor_type]
        rank = f.argsort().argsort() + 1  # low cancer fraction rank first

        cutoff_i = int(np.ceil(np.quantile(range(1, N+1), top_rank * 0.01)))
        selected = rank <= cutoff_i
        adata.obs[f'rank{top_rank}_pred'] = [
            'hijacking' if selected.iloc[i] else 'non-hijacking' 
            for i in range(N)
        ]
    

    def eval_pred(self, label_series, top_ranks=np.arange(100, -5, -5), probs=np.arange(100,-5, -5)):
        adata = self.mixture_cell_adata
        if not label_series.index.equals(adata.obs.index):
            raise ValueError('the index of label_series and mixture_cell_adata.obs.index are not identical')
        label_series = label_series.apply(lambda x: x == 'hijacking')
        label = label_series.astype(int)
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 12), dpi=300)
        axs = axs.flatten()
        
        colors = ['#3183be', '#4693c4', '#5aa3ca', '#6eb3cf', '#82c3d5', '#96d3db', '#aadfe1', '#bee7e6', '#d2f1ec', '#dfebf7']
        
        ax = axs[0]
        for top_rank in top_ranks:
            pred = adata.obs[f'rank{top_rank}_pred'].apply(lambda x: x == 'hijacking').astype(int)
            fpr, tpr, _ = roc_curve(label, pred, pos_label=1)
            auroc = roc_auc_score(label, pred)
            ax.plot(fpr, tpr, label=f"rank{top_rank}, AUROC={str(round(auroc, 2))}", color=colors[top_ranks.tolist().index(top_rank) % len(colors)])
        ax.set_title('Rank pred ROC curve')
        ax.legend(loc="lower right")
        ax.grid(False)
        
        ax = axs[1]
        for top_rank in top_ranks:
            pred = adata.obs[f'rank{top_rank}_pred'].apply(lambda x: x == 'hijacking').astype(int)
            fpr, tpr, _ = precision_recall_curve(label, pred, pos_label=1)
            auprc = average_precision_score(label, pred)
            ax.plot(fpr,tpr, label=f"rank{top_rank}, AUPRC={str(round(auprc, 2))}", color=colors[top_ranks.tolist().index(top_rank) % len(colors)])
        ax.set_title('Rank pred Precision-Recall curve')
        ax.legend(loc="lower left")
        ax.grid(False)
        
        ax = axs[2]
        for prob in probs:
            pred = adata.obs[f'prob{prob}_pred'].apply(lambda x: x == 'hijacking').astype(int)
            fpr, tpr, _ = roc_curve(label, pred, pos_label=1)
            auroc = roc_auc_score(label, pred)
            ax.plot(fpr, tpr, label=f"prob{prob}, AUROC={str(round(auroc, 2))}", color=colors[probs.tolist().index(prob) % len(colors)])
        ax.set_title('Prob pred ROC curve')
        ax.legend(loc="lower right")
        ax.grid(False)
        
        ax = axs[3]
        for prob in probs:
            pred = adata.obs[f'prob{prob}_pred'].apply(lambda x: x == 'hijacking').astype(int)
            precision, recall, _ = precision_recall_curve(label, pred, pos_label=1)
            auprc = average_precision_score(label, pred)
            ax.plot(recall, precision, label=f"prob{prob}, AUPRC={str(round(auprc, 2))}", color=colors[probs.tolist().index(prob) % len(colors)])
        ax.set_title('Prob pred Precision-Recall curve')
        ax.legend(loc="lower left")
        ax.grid(False)
        
        plt.tight_layout()
        plt.show()

    def calculate_metrics(self, label, pred):
        roc_auc = roc_auc_score(label, pred)

        tn, fp, fn, tp = confusion_matrix(label, pred).ravel()

        accuracy = (tp + tn) / (tp + fp + fn + tn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        sensitivity = recall
        specificity = tn / (tn + fp)
        ppv = precision
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1 = f1_score(label, pred)
        kappa = cohen_kappa_score(label, pred)
        brier = brier_score_loss(label, pred)

        avg_pr_auc = average_precision_score(label, pred)

        return {
            'AUROC': roc_auc,
            'AUPRC': avg_pr_auc,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'PPV': ppv,
            'NPV': npv,
            'F1_Score': f1,
            'Kappa': kappa,
            'Brier_Score': brier
        }   
        
    def get_result(self, label_series,type='Rank', top_ranks=np.arange(100, -5, -5)):
        adata = self.mixture_cell_adata
        if not label_series.index.equals(adata.obs.index):
            raise ValueError('the index of label_series and mixture_cell_adata.obs.index are not identical')
        label_series = label_series.apply(lambda x: x == 'hijacking')
        label = label_series.astype(int)

        results = pd.DataFrame()

        for top_rank in top_ranks:
            if type == 'Rank':
                pred = adata.obs[f'rank{top_rank}_pred'].apply(lambda x: x == 'hijacking').astype(int)
            elif type == 'Prob':
                pred = adata.obs[f'prob{top_rank}_pred'].apply(lambda x: x == 'hijacking').astype(int)
            elif type == 'Consensus(Interaction)':
                pred = ((adata.obs[f'rank{top_rank}_pred'] == 'hijacking') & 
                        (adata.obs[f'prob{top_rank}_pred'] == 'hijacking')).astype(int)
                
            elif type == 'Consensus(Union)':
                pred = ((adata.obs[f'prob{top_rank}_pred'] == 'hijacking') | 
                        (adata.obs[f'rank{top_rank}_pred'] == 'hijacking')).astype(int)
            else:
                raise ValueError("Invalid prediction type")
            
            metrics = self.calculate_metrics(label, pred)
            metrics[type] = top_rank 
            results = pd.concat([results, pd.DataFrame([metrics])], ignore_index=True)




        return results
     
    def plot(self, plots=None):
        available_plots = {
            "ref_profiles": lambda: self._plot_ref_profiles(),
            "mixture_cell": lambda: self._plot_heatmap(self.mixture_cell_adata.X, "Mixture Cell Profiles"),
            "U": lambda: self._plot_heatmap(self.U, "Predicted mean (U)"),
            "B": lambda: self._plot_heatmap(self.B, "Predicted variance (B)"),
            "raw_F": lambda: self._plot_heatmap(self.raw_F, "Predicted raw fraction"),
            "F": lambda: self._plot_heatmap(self.F, "Predicted relative fraction")
        }

        if plots is None:
            plots = available_plots.keys()  
        
        for plot_name in plots:
            if plot_name in available_plots:
                available_plots[plot_name]()
            else:
                print(f"[WARNING] '{plot_name}' is not a valid plot option. Available options: {list(available_plots.keys())}")

    def _plot_ref_profiles(self):
        for t, adata in self.ref_adata_dict.items():
            self._plot_heatmap(adata.X, f"Ref {t} Profiles")

    def _plot_heatmap(self, data, title):
        sns.heatmap(data)
        plt.title(title)
        plt.show()

        
