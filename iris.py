# IRIS object definition
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import anndata as ad
import scvi
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
import sklearn.metrics as skm
import seaborn as sns
import csv
import random
import time
import gseapy as gp
import scipy

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, decomposition

if TYPE_CHECKING:
    from anndata import AnnData
    from scvi.module import VAE
    from numpy.typing import NDArray

class IRIS:
    def __init__(
            self, 
            name: str, 
            signals: list[str] = [], 
            pathways: dict[str, str] = {}, 
            anndata: AnnData = None, 
            include_presets: bool = True
        ):
        '''
        Initializes IRIS object. 

        Args:
            name: string with name of IRIS object
            signals: list of strings of signals to analyze (ex. ["RA", "Wnt"])
            pathways: dictionary of form {signal1: [gene1, gene2, ..]}, signals and genes should all be strings
            anndata: AnnData object to initialize IRIS object with
            model: model to do IRIS analysis with 
            diffmaps: dictionary storing diffusion component data for various combinations of cell types
            include_presets: toggles inclusion of ["RA", "Bmp", "Fgf", "Wnt", "TgfB", "HH"] signals and pathways, default True
        '''
        self.name = name
        self.signals = signals
        self.pathways = pathways
        self.anndata = anndata 
        self.models = {}
        self.diffmaps = {} # dict of dict of anndata objects + extra info
        self.gene_weights = {}

        if include_presets:
            preset_signals, preset_mappings = self.make_presets()
            self.signals += preset_signals
            preset_mappings.update(pathways) # updates preset_mappings in place, writes over with inputted data
            self.pathways = preset_mappings
        
    def __str__(self):
        return f"IRIS object {self.name}"
    
    def make_presets(self) -> tuple[list[str], dict[str, str]]:
        '''
        Creates preset gene mapping with RA, Bmp, Fgf, Wnt, TgfB signals. 
        Returns:
            preset_signals: list of preset signals 
            mappings: dictionary of preset signals mapped to response genes
        '''
        preset_signals = ["RA", "Bmp", "Fgf", "Wnt", "TgfB"] # "HH"

        mappings = {}
        mappings["RA"] = ["CYP26A1", "HOXB1", "HNF1B", "CYP26C1", "HOXA1", "HOXB2", "HOXA3", "HOXA2", "HOXB3"]
        mappings["Bmp"] = ["ID2", "BMPER", "ID4", "ID1", "BAMBI", "MSX1", "ID3", "MSX2", "SMAD7"]
        mappings["Fgf"] = ["SPRY1", "SPRY2", "DUSP6", "SPRY4", "ETV5", "ETV4", "FOS", "SPRY2", "MYC", "JUNB", "DUSP14"]
        mappings["Wnt"] = ["AXIN1", "CCND1", "DKK1", "MYC", "NOTUM", "AXIN2", "SP5", "LEF1"]
        mappings["TgfB"] = ["SMAD7", "TWIST1", "TWIST2"]
        # mappings["HH"] = ["GLI1", "PTCH1", "HHIP", "FOXF1", "FOXF2"]

        for signal in preset_signals:
            genes = mappings[signal]
            self.gene_weights[signal] = {}
            for gene in genes:
                self.gene_weights[signal][gene] = 1 

        return preset_signals, mappings

    def conv_stim_to_code(
            self, 
            val: AnnData
        ) -> str:
        '''
        Creates string to represent experimental condition (ex. 'Bmp+TgfB+Fgf-Wnt+RA-' for 
        presence of Bmp, TgfB, and Wnt only). Takes data entry from AnnData object.

        Args:
            val: data to be converted to code

        Returns:
            out: string code representing condition
        '''
        out = ''
        for signal in self.signals:
            if val[signal + '_class'] == 'Stim':
                out += signal+'+'
            else:
                out += signal+'-'
        return out
    
    def add_pathway(
            self, 
            signal: str, 
            path: str
        ) -> None:
        '''
        Add a new signaling pathway with a new set of response genes to the IRIS object via 
        importing a csv file. Expects CSV to have two columns - "gene", weight.

        Args:
            signal: string of signal name associated with new pathway
            path: string of filepath to csv
        '''

        i = 0
        self.signals += [signal]
        self.pathways[signal] = []
        self.gene_weights[signal] = {}

        with open(path, mode='r') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                if i == 0:
                    i += 1
                    continue
                gene = line[0]
                self.pathways[signal] += [gene]
                self.gene_weights[signal][gene] = float(line[1:])

    def set_gene_weights(
            self, 
            signal: str, 
            weights: dict,
            normalize: bool
        ) -> None:
        '''
        Scales gene weights for a given signaling pathway, saves to IRIS.gene_weights. 
        Weights should be a dictionary of genes to values, with option to scale weights 
        to 1 (normalize parameter). 

        Args:
            signal: string of signal name 
            weights: dictionary of {(str) gene: (float/int) weight} given by user
            normalize: boolean whether to scale weights to max 1 or not
        '''
        self.gene_weights[signal].update(weights)

        if normalize:
            scale = 1/max(self.gene_weights[signal].values())
        else:
            scale = 1

        for gene in self.gene_weights[signal]:
            self.gene_weights[signal][gene] *= scale

    def response_gene(
            self, 
            batches: list[int]
        ) -> None:
        '''
        Calculates the response gene score on all response genes of all signals on normalized 
        or unnormalized data. Any user modification of weights should be done before calling 
        response_gene by calling set_gene_weights(). Modifies the IRIS object's anndata object 
        to include the response gene values under "AnnData.obs[SIGNAL_resp_zorn]".

        Args:
            batches: list of numbers of which batches to calculate response gene score with
        '''
        adata_gifford = self.anndata[np.isin(self.anndata.obs['batch'], batches)]
        sc.pp.normalize_total(adata_gifford, target_sum=1e4)
        sc.pp.log1p(adata_gifford)

        for signal in self.signals:
            name = signal + '_resp_zorn'
            resp_lst = self.pathways[signal]
            mat = adata_gifford[:, resp_lst].X.todense()

            # assuming mat columns are in same order as resp_lst
            for i in range(len(resp_lst)):
                gene = resp_lst[i]
                weight = self.gene_weights[signal][gene]
                mat[:,i] *= weight

            mat = (mat - np.mean(mat, axis=1))/np.std(mat, axis=1)
            mat[np.isnan(mat)] = 0
            lst = []
            for val in np.array(mat.sum(axis=1)):
                lst.append(val[0])
            adata_gifford.obs[name] = lst
            adata_gifford.obs[name] /= adata_gifford.obs[name].max()
            self.anndata.obs[name] = adata_gifford.obs[name]
    
    def generate_diffusion(
            self, 
            cell_types, 
            batches: list[int] = None, 
            stages: list[str] = None
        ) -> None:
        '''
        Calculates diffusion components for the given cell types. Plots diffusion map, stores
        subset of anndata object corresponding to this set of cell types in self.diffmaps.

        Args:
            name: string of name for this diff anndata object
            cell_types: list of strings of cell types (ex. ['Epiblast', 'Mixed mesoderm', ...])
            batches: list of integers of batches to use when calculating diffusion map (optional)
            stages: list of strings of developmental stages to include (optional)
        '''
        name = ''.join(cell_types)
        if batches:
            adata_gast = self.anndata[np.isin(self.anndata.obs['batch'], batches)]
        else:
            adata_gast = self.anndata

        lst = []
        for i in range(len(adata_gast.obs)):
            lst.append(self.conv_stim_to_code(adata_gast.obs.iloc[i]))
        adata_gast.obs['code'] = lst

        if stages:
            adata_sub = adata_gast[np.isin(adata_gast.obs['celltype'], cell_types) & np.isin(adata_gast.obs['stage'], stages),:]
        else:
            adata_sub = adata_gast[np.isin(adata_gast.obs['celltype'], cell_types),:]

        sc.pp.normalize_total(adata_sub, target_sum=1e4)
        sc.pp.log1p(adata_sub)
        sc.pp.highly_variable_genes(adata_sub, min_mean=0.0125, max_mean=3, min_disp=0.5)

        subsets = []
        for kind in cell_types:
            subset = adata_sub[adata_sub.obs['celltype'] == kind]
            adata_i = sc.pp.subsample(subset, fraction=min(1, 4000/len(subset)), copy=True)
            subsets.append(adata_i)
        adata_diff = ad.concat(subsets)

        # make pca components
        sc.tl.pca(adata_diff, n_comps=50)
        sc.pp.neighbors(adata_diff, n_neighbors=100, n_pcs=50, use_rep='X_pca', method='gauss')
        sc.tl.diffmap(adata_diff, n_comps=10)
        # plot diffusion map
        sc.pl.diffmap(adata_diff, components=['2, 4'], projection='2d', color='celltype')

        self.diffmaps[name] = {'data': adata_diff} 

    def get_components(
            self, 
            trajectory: list[str], 
            subsets: list[NDArray], 
            n_comps: int
        ) -> list[int]:
        '''
        Finds the diffusion components that properly order the given trajectory given AnnData. 

        Args:
            trajectory: list of strings of cell types, in order
            subsets: list of ['X_diffmap'] data corresponding to the cell types in order
            n_comps: number of relevant diffusion components

        Returns:
            components: list of integers of diffusion components that properly order trajectory
        '''
        components = []
        n = len(trajectory)

        for i in range(1, n_comps): # relevant diffusion components start at index 1
            front = True
            back = True
            forward = []
            reverse = []
            for cell_type in trajectory:
                diff_component = subsets[cell_type][:,i]
                avg_resp = np.mean(diff_component)
                # collect values along diffusion component
                forward.append(avg_resp)
                # collect values along reverse of diffusion component
                reverse.append(-1*avg_resp)
            for j in range(1, n):
                # diffusion component incorrectly orders
                if forward[j] < forward[j-1]:
                    front = False
                if reverse[j] < reverse[j-1]:
                    back = False
            # diffusion component orders correctly in either forward or reverse
            if front or back: 
                components.append(i+1)

        return components
        
    def select_diffusion_components(
            self, 
            trajectory: list[str]
        ) -> tuple[int, list[int]]:``
        '''
        For a set of cell type names finds the component on the diffusion maps which best 
        separate the cell types in the correct order and saves it for future analysis.
        Stores best diffusion component and all properly-ordering components in self.diffmaps
        with the anndata object that originated it.
        
        Args:
            trajectory: list of strings of cell types in order

        Returns:
            best_dc: best diffusion component that correctly orders into trajectory; best = widest range
                Returns -1 if no diffusion components identified 
            components: list of numbers representing all diffusion components (1 = DC1, 2 = DC2, etc); 
                Returns [] if no components identified
        ''' 
        long_name = ''.join(trajectory)
        # TODO: consider making one diffmap once for IRIS object that has all celltypes, and getting 
        # components from that diffmap rather than making multiple diffusion maps depending on the 
        # trajectories being asked for
        
        # THINK: maybe long-term only having one diffusion map computer once for entire anndata object 
        # is way to go, so should never have to recompute after self.diffmaps is non-empty / not None
            # compute diffusion map once with all cell types that are even in the data
            # probably separate out and have self.diffmap as its own attribute corresponding to anndata object
                # after running diffmap, and all of this checking logic below would just be checking
                # if self.diffmap already exists or not

        # check if best components already found or adata obj w diffmap already exists
        exists = False
        for diffmap in self.diffmaps:
            has_all = True
            for cell_type in trajectory:
                if cell_type not in diffmap:
                    has_all = False
                    break
            if has_all:
                exists = True
                # have already found best dc for this trajectory, shortcut and return
                if long_name in self.diffmaps[diffmap]:
                    return self.diffmaps[diffmap][long_name]['best_dc'], self.diffmaps[diffmap][long_name]['components']
                # haven't found component, but diffmap contains all cell types we need
                adata = self.diffmaps[diffmap]
                break
                
        if not exists:
            self.generate_diffusion(trajectory)
            adata = self.diffmaps[long_name]
        
        data = adata['data']
        subsets = {}

        # make subsets of all cell types once
        for cell_type in trajectory:
            subsets[cell_type] = data[data.obs['celltype'] == cell_type].obsm['X_diffmap']

        n_comps = data.obsm['X_diffmap'].shape[1]
        diff_components = self.get_components(trajectory, subsets, n_comps)
            
        widest = 0
        best_dc = -1

        if diff_components:
            # find best DC
            for c in diff_components:
                lowest_val = np.mean(subsets[trajectory[0]][:,c-1])
                highest_val = np.mean(subsets[trajectory[-1]][:,c-1])
                span = abs(highest_val - lowest_val)
                if span > widest:
                    widest = span
                    best_dc = c

            if long_name not in adata:
                adata[long_name] = {}
                
            adata[long_name]['components'] = diff_components
            adata[long_name]['best_dc'] = best_dc

        return best_dc, diff_components
    
    def set_scvi_model(
            self, 
            data: AnnData, 
            n_layers: int = 2, 
            n_latent: int = 30, 
            n_hidden: int = 128, 
            epochs: int = 10
        ) -> VAE:
        '''
        Initializes SCVI model to use for IRIS analysis, returns model.

        Args:
            data: data to set up model on, passed into setup_anndata
            n_layers: number of layers model should have (default 2)
            n_latent: dimensionality of latent space (default 30)
            n_hidden: number of hidden nodes (default 128)
            epochs: maximum epochs to train SCVI model (default 10)

        Returns:
            vae: SCVI model with given params set up on given data
        '''
        # adds fields to anndata
        scvi.model.SCVI.setup_anndata(data, layer="counts", batch_key='batch', categorical_covariate_keys=['celltype'])
        # sets up model with this anndata
        vae = scvi.model.SCVI(data, n_layers=n_layers, n_latent=n_latent, n_hidden=n_hidden, gene_likelihood="zinb")
        vae.train(max_epochs=epochs, use_gpu=True)
        return vae
        
    def run_model(
            self, 
            train_batches: list[int], 
            test_batches: list[int], 
            out_path: str, 
            n_layers: int = 2, 
            n_latent: int = 30
        ) -> None: # TODO: consider allowing control of vae epochs and scanvi epochs too
        '''
        Creates and runs VAE model contained in IRIS object with given hyperparameters. Model 
        runs with random 80/20 train/test split across all batches and conditions. Stores final 
        trained models for each signal into models attribute of the IRIS object (IRIS.models). 
        Saves model predictions into csv at outpath.
        
        Args:
            train_batches: list of integers of batches to obscure while training 
            test_batches: list of integers of batches to keep labels for
            out_path: string of filename to write results to
            n_layers: number of hidden layers, default 2
            n_latent: dimensionality of latent space, default 30
        '''
        self.anndata.obs['Clusters'] = "unknown"

        adata_full_gifford = self.anndata[np.isin(self.anndata.obs['batch'], train_batches + test_batches)]
    
        held_out = np.random.choice(adata_full_gifford.obs.index, size=int(len(adata_full_gifford.obs.index)*0.2))
        
        df_val = adata_full_gifford.obs.loc[held_out, :].copy()
        
        adata_full_gifford.layers['counts'] = adata_full_gifford.X
        adata_full_gifford = adata_full_gifford.copy()
        
        adata_full_gifford.obs['celltype'] = adata_full_gifford.obs['celltype'].cat.add_categories('unknown')
        adata_full_gifford.obs.loc[adata_full_gifford.obs[np.isin(adata_full_gifford.obs['batch'], train_batches)].index, 'celltype'] = 'unknown'

        vae = self.set_scvi_model(adata_full_gifford, n_layers=n_layers, n_latent=n_latent, epochs=100)

        df = pd.DataFrame({}, index=adata_full_gifford[held_out].obs.index)

        class_names = []
        for signal in self.signals:
            class_names.append(signal + '_class')
            adata_full_gifford.obs.loc[held_out, signal + '_class'] = "unknown"

        for val in class_names:
            scanvae = scvi.model.SCANVI.from_scvi_model(vae, labels_key=val, unlabeled_category = "unknown")
            scanvae.train(max_epochs=10)
            df[val] = scanvae.predict(adata_full_gifford[held_out], soft=True)['Stim'].values
            self.models[val] = scanvae
        
        df.to_csv(out_path)
    
    def validate_adata(
            self, 
            classes: list[str]
        ) -> bool:
        '''
        Makes sure that the given classes columns of the IRIS Anndata object only have 
        "Stim" and "Ctrl" values so that class predictions can be made. Prints errors 
        if Anndata object does not follow this format.

        Args:
            classes: list of strings of prediction classes to check for 
                (ex. ["Bmp_class_orig", "RA_class_orig"])

        Returns: 
            result: True if adata is in correct form, False if adata is not
        '''
        for c in classes:
            try:
                vals = self.anndata.obs[c]
                if all(val in ["Stim", "Ctrl"] for val in vals):
                    return True
                else:
                    print(f"prediction class labels are not categorical with [Stim, Ctrl] for class {c}")
                    return False
            except:
                print(f"anndata object missing column {c}")
                return False
                
    def evaluate(
            self, 
            batches: list[int],
            signals: list[str] = None, 
            metrics: list[str] = None, 
            plot: bool = True
        ) -> dict[str, float]: 
        '''
        Takes the predictions from the model and returns whichever statistic is asked for - 
        an AUPRC, AUROC curve or F1 score. If no pathway or metric is given, calculates all 
        metrics for all pathways. Stores final trained models for each signal into models 
        attribute of the IRIS object (IRIS.models).
        
        Args:
            signals: list of strings of signals to calculate score on (ex. "Wnt"); if not given, 
                all signals are used
            metrics: list of strings of desired statistics; if not given, all metrics are computed
            plot: whether or not to display AUROC, AUPRC curves (default True)

        Returns:
            scores: dictionary of dictionaries where keys are signals (RA, WNT, etc.) and values are dictionaries
            mapping metric to calculated scores
        '''
        # TODO: i fear i misunderstood this function when i first wrote it. now i think it should take in 
        # a csv or dataframe of predictions and run these metrics / plots on the predictions and corresponding
        # data, rather than initialize and run model again, then evaluate and plot (how this currently is). that 
        # way, this function can be a more generalized evaluater/plotter for any set of predictions, even though
        # the other functions (ex. held_out, cross_validate) have their own evaluation and plotting logic self-
        # contained (maybe they shouldnt?). if my thoughts are correct, then i will fix soon. 

        scores = {}

        if not signals:
            signals = self.signals
        if not metrics:
            metrics = ["AUROC", "F1", "AUPRC"]

        self.response_gene(batches)
        data = self.anndata[np.isin(self.anndata.obs['batch'], batches)]
        data.layers['counts'] = data.X
        df = pd.DataFrame({}, index=data.obs.index)
                        
        # all lists of lists
        if "AUROC" in metrics:
            auroc_tprs = []
            auroc_fprs = []
            resp_genes_tprs = []
            resp_genes_fprs = []
        if "AUPRC" in metrics:
            auprc_precisions = []
            auprc_recalls = []
            resp_genes_precisions = []
            resp_genes_recalls = []

        for signal in signals:
            class_name = signal + '_class'
            name = signal + '_resp_zorn'
            scores[signal] = {}

            vae = self.set_scvi_model(data, epochs=400)
            scanvae = scvi.model.SCANVI.from_scvi_model(vae, labels_key=class_name, unlabeled_category = "unknown")
            scanvae.train(max_epochs=5, batch_size=512)
            self.models[class_name] = scanvae

            df[class_name] = scanvae.predict()
            threshold = self.find_optimal_cutoff((data.obs[class_name] == "Stim").values.astype(int), data.obs[name])

            for metric in metrics:
                if metric == "AUROC":
                    score = skm.roc_auc_score(data.obs[class_name], df[class_name], pos_label='Stim')
                    tpr, fpr, _ = skm.roc_curve(data.obs[class_name], df[class_name], pos_label='Stim')
                    auroc_tprs.append(tpr)
                    auroc_fprs.append(fpr)

                    tpr, fpr, _ = skm.roc_curve((data.obs[class_name] == "Stim").astype(int), (data.obs[name].values > threshold).astype(int))
                    resp_genes_tprs.append(tpr)
                    resp_genes_fprs.append(fpr)
                elif metric == "F1":
                    score = skm.f1_score(data.obs[class_name], df[class_name], pos_label='Stim')
                elif metric == "AUPRC":
                    precision, recall, _ = skm.precision_recall_curve(data.obs[class_name], df[class_name], pos_label='Stim')
                    score = skm.auc(recall, precision)
                    auprc_precisions.append(precision)
                    auprc_recalls.append(recall)

                    precisions, recalls, _ = skm.precision_recall_curve((data.obs[class_name] == "Stim").astype(int), (data.obs[name].values > threshold).astype(int))
                    resp_genes_precisions.append(precisions)
                    resp_genes_recalls.append(recalls)
                else:
                    print(f"Metric {metric} is not supported")
                    break

                scores[signal][metric] = score

        # TODO: think this plotting code could be less repetitive
        if plot:
            colors = ['#D62728', '#17BECF', '#2CA02C', 'black', '#8C564B']
            if auroc_tprs:
                f = plt.figure(1)
                for i in range(len(auroc_tprs)):
                    plt.plot(resp_genes_tprs[i], resp_genes_fprs[i], color=colors[i])
                    plt.plot(auroc_tprs[i], auroc_fprs[i], color=colors[i], linestyle='dashed', dashes=(5, 5))
                    plt.xlim([1e-6, 1])
                    plt.ylim([0, 1])
                plt.legend(self.signals + ['IRIS', 'Reponse Genes'])
                plt.ylabel('True Positive Rate')
                plt.xlabel('False Positive Rate')
                plt.title('AUROC Signaling Regressed Prediction')

            if auprc_precisions:
                g = plt.figure(2)
                for i in range(len(auprc_precisions)):
                    plt.plot(resp_genes_recalls[i], resp_genes_precisions[i], color=colors[i])
                    plt.plot(auprc_recalls[i], auprc_precisions[i], color=colors[i], linestyle='dashed', dashes=(5, 5))
                    plt.xlim([1e-6, 1])
                    plt.ylim([0, 1])
                plt.legend(self.signals + ['IRIS', 'Reponse Genes'])
                plt.ylabel('Precision')
                plt.xlabel('Recall')
                plt.title('Precision-Recall Signaling Prediction')
            
            plt.show()

        return scores

    def find_optimal_cutoff(
            self, 
            target, 
            predicted
        ) -> list:
        '''
        Helper function to find optimal threshold of response gene value to be considered 
        positive indicator of condition. Used in IRIS.held_out_condition_validation to 
        score response gene value method. 

        Args:
            target: array-like true values 
            predicted: array-like predicted values

        Returns:
            list of threshold values
        '''
        fpr, tpr, threshold = skm.roc_curve(target, predicted)
        i = np.arange(len(tpr)) 
        roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
        roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

        return list(roc_t['threshold']) 
    
    def held_out_condition_validation(
            self, 
            batches: list[int], 
            condition: str = None, 
            category: str = None, 
            groupings: list[str] = None, 
            plot_each_condition: bool = False
        ) -> tuple[list[float], list[float]]:
        '''
        Runs the model excluding the given condition (ex.'BMP+TGFB+FGF-WNT+RA-') and 
        returns F1 score. Can also hold out condition across certain groupings of a 
        given category (cell type, batch, etc.). Saves final trained models for each 
        signal into models attribute of the IRIS object (IRIS.models).
        
        Args:
            batches: list of integers representing which batches to use 
            condition: string of condition to be held out (ex. BMP+TGFB+FGF-WNT+RA-); if None, 
                holds out each condition in the data one by one
            category: categorical variable to hold out (ex. cell type, batch, etc.)
            groupings: list of lists of categorical values to hold out (ex. ["Epiblast", 
                "Primitive Streak"], ["Gut"]..)

        Returns:
            lst_nn_score: F1 score of model predictions
            lst_resp_score: F1 score using response gene values and thresholding
        '''
        if category and not groupings:
            print("if category provided, must provide a grouping of category values as well")
            return -1, -1

        self.anndata.obs['Clusters'] = "unknown"
        self.response_gene(batches)

        adata_full_gifford = self.anndata[np.isin(self.anndata.obs['batch'], batches)]
        adata_full_gifford.layers['counts'] = adata_full_gifford.X
        adata_full_gifford = adata_full_gifford.copy()

        class_names = []
        for signal in self.signals:
            class_names.append(signal + '_class')
        
        # create codes for each entry
        code_lst = []
        for val in range(len(adata_full_gifford.obs)):
            code_lst.append(self.conv_stim_to_code(adata_full_gifford.obs.iloc[val]))
        adata_full_gifford.obs['code'] = code_lst
            
        # generate combinations to test
        if not condition:
            combos = adata_full_gifford.obs['code'].value_counts().index
        elif len(condition) == 1:
            combos = [condition]
        else:
            combos = condition 

        lst_resp = []
        lst_nn_score = []

        if not category:
            groupings = [0]

        for grouping in groupings:
            count = 0
            for combo in combos:
                count += 1
                # process data based on category, groupings, combos
                if category:
                    adata_non_cat = adata_full_gifford[~np.isin(adata_full_gifford.obs[category], grouping)]
                    adata_cat = adata_full_gifford[np.isin(adata_full_gifford.obs[category], grouping)]
                else:
                    adata_cat = adata_full_gifford
        
                adata_cat_out = adata_cat[np.isin(adata_cat.obs['code'], combo)]
                adata_cat_in = adata_cat[~np.isin(adata_cat.obs['code'], combo)]
                # true data values
                results = adata_cat_out.obs[class_names].values
                adata_cat_out.obs[class_names] = 'unknown'
                adata_full_giff2 = ad.concat([adata_non_cat, adata_cat_in, adata_cat_out])
                out_index = (adata_cat_out.obs.index)
                df = pd.DataFrame({}, index=adata_full_giff2.obs.index)

                # initialize VAE
                vae = self.set_scvi_model(adata_full_giff2, epochs=25)

                i = 0
                for classification in class_names:
                    if len(results[:, i]) == 0:
                        continue
                    # train SCANVI model
                    scanvae = scvi.model.SCANVI.from_scvi_model(vae, labels_key=classification, unlabeled_category = "unknown")
                    scanvae.train(max_epochs=5, batch_size=512)
                    self.models[classification] = scanvae
                    # make predictions
                    df[classification] = scanvae.predict()
                    out_results = df[np.isin(adata_full_giff2.obs.index, out_index)]
                
                    threshold = self.find_optimal_cutoff((adata_cat_in.obs[classification] == "Stim").values.astype(int), adata_cat_in.obs[classification.split('_')[0] + '_resp_zorn'])
                    
                    if (results[0, i]) == "Stim":
                        lst_nn_score.append(skm.f1_score(results[:, i], out_results[classification], pos_label='Stim'))
                        lst_resp.append(skm.f1_score((results[:, i] == "Stim").astype(int), (adata_cat_out.obs[classification.split('_')[0] + '_resp_zorn'].values > threshold).astype(int)))
                    i += 1

                # plot each condition within each grouping
                if plot_each_condition:
                    plt.figure()
                    plt.scatter(lst_resp, lst_nn_score, c=['#EB2027', '#29ABE2', '#00A64F', '#A87C4F', '#231F20'])
                    plt.axline((0, 0), slope=1, color='k', ls='--')
                    plt.xlabel('Response Gene F1 score')
                    plt.ylabel('IRIS F1 score')
                    plt.title(grouping+', '+combo)
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.show()

            # plot per grouping, aggregating all conditions
            plt.figure()
            plt.scatter(lst_resp, lst_nn_score, c=['#EB2027', '#29ABE2', '#00A64F', '#A87C4F', '#231F20'] * count)
            plt.axline((0, 0), slope=1, color='k', ls='--')
            plt.xlabel('Response Gene F1 score')
            plt.ylabel('IRIS F1 score')
            plt.title(grouping)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.show()
                    
        return lst_nn_score, lst_resp
    
    def generate_bitflip(
            self, 
            n: int, 
            references: list[str]
        ) -> list:
        '''
        Helper function that flips a string of 0's and 1's one-by-one and returns list of 
        generated sequences. For example, given '101', this function will return 
        ['001', '111', '100']. Supports multiple reference strings. 

        Args:
            n: integer number of sequences to generate
            references: list of strings of reference sequences

        Returns:
            sequences: list of strings of generated sequences
        '''
        sequences = []
        l = len(references)
        count = 0

        for i in range(l):
            combos = []
            reference = references[i]
            m = len(reference)
            for j in range(m):
                flipped = reference[:j] + str(1-int(reference[j])) + reference[j+1:]
                combos.append(flipped)
                count += 1
                # generated enough from bitflip
                if count == n:
                    sequences += combos
                    return sequences
                
            sequences += combos
        return sequences

    def hamming_distance(
            self, 
            string1: str, 
            string2: str
        ) -> int:
        '''
        Helper function that finds Hamming distance between two strings (sums up 
        number of characters that differ). Both strings must be same length.

        Args:
            string1: first string 
            string2: second string

        Returns:
            dist: integer Hamming distance
        '''
        dist = 0
        for i in range(len(string1)):
            if string1[i] != string2[i]:
                dist += 1
        return dist

    def generate_random(
            self, 
            n: int, 
            reference: str, 
            min_distance: int = None, 
            sampling: bool = False
        ) -> set:
        '''
        Generates n bitstrings that are each min_distance away from all other 
        generated bitstrings, using reference as the starting point. If min_distance 
        is not provided, samples increasing distances to find minimum Hamming 
        distance able to generate n bitstrings. Uses timeout of 30s while sampling.

        Args:
            n: integer number of bitstrings to be generated
            reference: string, reference bit sequence (ex. '0110')
            min_distance: integer minimum distance apart each bitstring should be 
            sampling: boolean of whether to sample minimum distance or not 

        Returns:
            combos: set of generated bitstrings, including reference 
        '''
        length = len(reference)

        if not min_distance:
            min_distance = 1
            sampling = True

        valid = False
        i = 0
        combos = {reference}
        num_to_permute = min_distance

        start = time.time()
        while i < n:
            indices = random.sample(range(0,length), num_to_permute)
            perturbed = ''.join([str(1-int(reference[i])) if i in indices else reference[i] for i in range(length)])
            if perturbed not in combos:
                for elem in combos:
                    if self.hamming_distance(elem, perturbed) < min_distance:
                        valid = False
                        break
                    valid = True
                if valid:
                    combos.add(perturbed)
                    i += 1
            curr = time.time()

            while curr - start > 30:
                if num_to_permute < length:
                    num_to_permute += 1
                    start = time.time()
                    break
                if sampling:
                    # this distance timed out, so use results from previous one
                    return self.generate_random(n, reference, min_distance - 1, False)
                else:
                    # process timed out with a defined minimum distance
                    return combos
            
        if sampling and min_distance < length:
            # keep looking only if distance away isn't maxed out 
            min_distance += 1
            return self.generate_random(n, reference, min_distance, True)

        return combos

    def generate_combinations(
            self, 
            num: int, 
            frac_random: float, 
            reference: str, 
            distance: int = None
        ) -> list[str]:
        '''
        Generates num sequences of bitstrings using random and/or bitflip technique, 
        based on provided reference string (ex. '0110'). Random technique generates 
        bitstrings that are at least [distance] apart, defined by Hamming distance. 
        Bitflip technique flips each bit of the reference bitstring. Returns list of 
        generated bitstrings.

        Args:
            num: integer number of sequences to generate
            frac_random: float of proportion to be generated via random technique
            reference: reference bitstring (ex. '0110')
            distance: float minimum distance apart randomly-generated strings should be

        Returns:
            list of generated bitstrings
        '''
        num_random = round(num * frac_random)
        random_combos = list(self.generate_random(num_random, reference, distance))

        num_bitflip = num - num_random
        # assuming only one reference sequence to bitflip off of, but could be more
        bitflip_combos = self.generate_bitflip(num_bitflip, [reference])

        return random_combos + bitflip_combos

    def scanpy_recipe(
            self,
            batches: list[int],
            num_genes: int = 5
        ) -> None:
        '''
        Performs basic PCA, Leiden clustering, and differential gene expression analysis 
        on the IRIS object's data. Plots the num_genes most differentially expressed genes.

        Args:
            batches: list of integers of batches to use for clustering
            num_genes: integer number of genes to plot for differential gene expr. analysis
        '''
        adata = self.anndata[np.isin(self.anndata.obs['batch'], batches)]
        adata = adata[~(adata.obs['celltype'] == 'unknown')]

        sc.tl.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        # clustering
        # sc.tl.leiden(self.anndata, flavor="igraph", n_iterations=2) # <- flavor for scanpy >1.10
        sc.tl.leiden(adata, n_iterations=2)

        colors = ['red', 'orange', 'green', 'blue', 'purple']
        i = 0
        for signal in self.signals:
            sc.pl.umap(adata, color=signal+'_class', palette=[colors[i], 'gray']) 
            i += 1
            
        # differential gene expression analysis
        sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")
        sc.pl.rank_genes_groups_dotplot(
            adata, groupby="leiden", standard_scale="var", n_genes=num_genes
        )

    def get_hyperparameter_score(
            self, 
            train_batches: list[int], 
            test_batches: list[int], 
            parameters: list[int], 
            signal: str, 
            metrics: list[str], 
            resample_labels: bool = False
        ) -> dict[str, float]:
        '''
        Creates model with given hyperparameters, makes predictions, then scores predictions 
        on the given metrics. Can also randomize the training labels before predicting on the 
        test data. Returns a dictionary of scores for each given metric.

        Args:
            train_batches: list of integers of batches used for training
            test_batches: list of integers of batches used for testing
            parameters: list of list of integers of parameter values
            signal: string of signal, ex. "RA"
            metrics: list of strings of metric, ex. ["AUROC"], ["AUROC", "F1"]
            resample_labels: boolean of whether to randomize training labels, default False
        '''
        hidden, layers, latent, epochs = parameters

        all_batches = train_batches + list(set(test_batches) - set(train_batches))
        adata_full = self.anndata[np.isin(self.anndata.obs['batch'], all_batches)]
        adata_full.layers['counts'] = adata_full.X
        adata_full = adata_full.copy()
        adata_in = adata_full[np.isin(adata_full.obs['batch'], train_batches)]
        adata_out = adata_full[np.isin(adata_full.obs['batch'], test_batches)]

        for signal in self.signals:
            adata_out.obs[signal+'_class'] = 'unknown'
            if resample_labels:
                adata_in.obs[signal+'_class'] = np.random.choice(['Stim', 'Ctrl'], len(adata_in))

        # adata_out.obs['celltype'] = 'unknown' # not sure if obscuring cell type too is desired

        adata = ad.concat([adata_in, adata_out])

        # original data 
        df_result = adata_full.obs[[signal+'_class' for signal in self.signals]]

        # suffix relates to which batches in/out
        df = pd.DataFrame({}, index=adata.obs.index)
        suffix = '_in_'+str(train_batches)+'_out_'+str(test_batches)

        vae = self.set_scvi_model(adata, layers, latent, hidden, 175) 
        class_name = signal + '_class'
        outfile_name = 'vae_scanvi_layers_' + str(int(layers)) + '_hidden=' + str(int(hidden)) + '_latent=' + str(int(latent)) + '_' + class_name + '_' + suffix + '_rs.csv'

        for i in range(3):
            vae.train(max_epochs=25)
            scanvae = scvi.model.SCANVI.from_scvi_model(vae, labels_key=class_name, unlabeled_category="unknown")
            for epoch in range(3):
                scanvae.train(max_epochs=epochs)

                df['Stim_' + class_name + '_scanvi=' + str(int(epoch)*3) + '_vae=' + str(int(i+7)*25)] = scanvae.predict(soft=True)['Stim']
            
                dir_path_a = 'vae_scanvi_layers_' + str(int(layers)) + '_hidden=' + str(int(hidden)) + '_latent=' + str(int(latent)) + '_' + class_name + '_rs/'
                dir_path = 'Stim_' + class_name + '_scanvi=' + str(int(epoch)*3) + '_vae=' + str(int(i+7)*25)
                final_dir_path = dir_path_a + dir_path + suffix
            
                scanvae.save(final_dir_path, overwrite=True) 

        df.to_csv(outfile_name)

        self.response_gene(all_batches)

        scores = {}
        df_result_sub = df_result[np.isin(adata_full.obs['batch'], test_batches)] # original values
        
        signal_lst = scanvae.predict(soft=True)['Stim'] # predictions
        signal_lst_sub = signal_lst[np.isin(adata.obs['batch'], test_batches)]

        for metric in metrics:
            if metric == "AUROC":
                score = skm.roc_auc_score(((df_result_sub[class_name]) == 'Stim').astype(int), signal_lst_sub)
            elif metric == "F1":
                score = skm.f1_score((df_result_sub[class_name] == "Stim").astype(int), signal_lst_sub)
            elif metric == "AUPRC":
                precision, recall, _ = skm.precision_recall_curve(((df_result_sub[class_name]) == 'Stim').astype(int), signal_lst_sub)
                score = skm.auc(recall, precision)
            else:
                print(f"Metric {metric} is not supported")
                break

            if metric not in scores:
                scores[metric] = [score]

        return scores

    def select_hyperparameters(
            self, 
            train_batches: list[int], 
            test_batches: list[int], 
            parameters: list[list[int]], 
            signal: str, 
            metrics: list[str], 
            resample_labels: bool = False
        ) -> tuple[dict[str, [list[int]]], dict[str, NDArray[np.float64]]]:
        '''
        Samples the best combination of hyperparameters based on given parameter values. 
        Evaluates model on training data based on given metric, returns best hyperparameter 
        combination and scores for each hyperparameter combination in numpy array. 

        Args:
            train_batches: list of integers of batches used for training
            test_batches: list of integers of batches used for testing
            parameters: list of list of integers of parameter values
            signal: string of signal, ex. "RA"
            metric: list of strings of metrics, ex. ["AUROC"]
            resample_labels: boolean of whether to randomize training labels, default False
        '''
        hidden, num_layers, latent_dim, epochs = parameters
        final_scores = {metric: np.zeros((len(hidden), len(num_layers), len(latent_dim), len(epochs))) for metric in metrics}

        best_scores = {metric: 0 for metric in metrics}
        best_params = {metric: None for metric in metrics}

        i = 0
        for hnode in hidden: 
            j = 0
            for nlayers in num_layers:
                k = 0
                for nlat in latent_dim: 
                    l = 0
                    for nepoch in epochs:
                        params = [hnode, nlayers, nlat, nepoch]
                        scores = self.get_hyperparameter_score(train_batches, test_batches, params, signal, metrics, resample_labels)
                        for metric in metrics:
                            score = scores[metric]
                            if score > best_scores[metric]:
                                best_scores[metric] = score
                                best_params[metric] = parameters
                            final_scores[metric][i,j,k,l] = score
                        l += 1
                    k += 1
                j += 1
            i += 1
        
        return best_params, final_scores
    
    def plot_hyperparameters(
            self, 
            scores: NDArray[np.float64], 
            dim1: int, 
            dim2: int, 
            test_batches: list[int], 
            signal: str, 
            parameters: list[list[int]] = None, 
            savefig: bool = True
        ) -> None:
        '''
        Takes final_scores from IRIS.select_hyperparameters and plots two given parameters 
        against each other in a heatmap. Takes parameters to plot as integers dim1, dim2. 
        0 = # hidden nodes, 1 = # layers, 2 = latent dimension size, 3 = # epochs. Saves 
        heatmap as .svg if desired.

        Args:
            scores: numpy NDArray of scores from select_hyperparameters
            dim1: integer representing first parameter
            dim2: integer representing second parameter
            test_batches: list of integers of batches used for testing
            signal: string of signal ex. "RA"
            parameters: list of list of integers of parameter values
            savefig: boolean of whether to save heatmap; default True
        '''
        suffix = ''.join([str(num) for num in test_batches])
        mapping = {0: "hidden_nodes", 1: "num_layers", 2: "latent_dim", 3: "epochs"}

        # swap axes to bring desired dimensions forward
        a2 = np.swapaxes(scores, 0, dim1)
        array = np.swapaxes(a2, 1, dim2)
        range1 = scores.shape[dim1]
        range2 = scores.shape[dim2]

        # average values along first two axes always
        list1 = []
        error_list1 = []
        for x in range(range1):
            list2 = []
            error_list2 = []
            for y in range(range2):
                val = np.mean(array[x, y, :, :])
                error = np.std(array[x, y, :, :])
                list2.append(val)
                error_list2.append(error)
            list1.append(list2)
            error_list1.append(error_list2)

        # plot heatmap
        ax = sns.heatmap(list1, annot=list1, annot_kws={'va':'bottom'})
        ax = sns.heatmap(list1, annot=error_list1, annot_kws={'va':'top'}, cbar=False)
        ax.set(xlabel=mapping[dim2], ylabel=mapping[dim1])
        desc = mapping[dim1] + '_' + mapping[dim2] # make informative file name 

        if parameters:
            ax.set_xticklabels(parameters[dim2])
            ax.set_yticklabels(parameters[dim1])
    
        if savefig:
            plt.savefig(signal + ' parameters ' + desc + suffix + '.svg')
    
    def randomize_training_labels(
            self, 
            train_batches: list[int], 
            test_batches: list[int], 
            parameters: list[list[int]], 
            metrics: list[str], 
            outpath: str
        ) -> tuple[dict, dict, dict]:
        '''
        Randomizes labels in the training data, fits model with all possible parameter 
        combinations and reruns model prediction. Calculates AUROC and/or AUPRC of model 
        predictions and plots ECDF of each signal over given metrics. Returns best 
        hyperparameters on random/true labels, scores on randomized labels, and scores on 
        true labels.

        Args:
            train_batches: list of integers of batches to use for training
            test_batches: list of integers of batches to use for testing
            parameters: list of list of integers in order [hidden nodes, num layers, 
                        latent dim, epochs]
            metrics: list of strings ex. ["AUROC"]; ["AUROC", "AUPRC"]
            outpath: string of filename to save figure(s) to

        Returns:
            best_params: dictionary of signal 
            random_scores: dictionary of best parameters for given signal+metric+randomization
            true_scores: dictionary of best parameters for given signal+metric+true labels
        '''
        best_params = {}
        random_scores = {}
        true_scores = {}

        for signal in self.signals:
            params, true_score = self.select_hyperparameters(train_batches, test_batches, parameters, signal, metrics, resample_labels=False)
            if signal not in true_scores:
                true_scores[signal] = {}
            for metric in metrics:
                true_scores[signal][metric] = true_score[metric].flatten()
                best_params[signal][metric]['true'] = params[metric]

            params, random_score = self.select_hyperparameters(train_batches, test_batches, parameters, signal, metrics, resample_labels=True)
            if signal not in random_scores:
                random_scores[signal] = {}
            for metric in metrics:
                random_scores[signal][metric] = random_score[metric].flatten()
                best_params[signal]['random'] = params[metric]

        j = 1
        for metric in metrics:
            plt.figure(j)
            colors = ['#D62728', '#17BECF', '#2CA02C', 'black', '#8C564B']
            i = 0
            for signal in self.signals:
                sns.kdeplot(random_scores[signal][metric], bw_adjust=0.1, cumulative=True, color=colors[i], linestyle='--')
                sns.kdeplot(true_scores[signal][metric], bw_adjust=0.1, cumulative=True, color=colors[i], linestyle='-')
                i += 1
            j += 1
            plt.legend(self.signals)
            plt.xlabel(metric)
            plt.ylabel("ECDF")

            # Save the plot
            plt.savefig(f"{outpath}_{metric}.png")
            plt.show()

        return best_params, random_scores, true_scores

    def cross_validate_batches(
            self, 
            train_batches: list[int], 
            validation_batches: list[int], 
            metric: str
        ) -> None:
        '''
        Performs cross validation holding out a list of batches and estimates given 
        performance metric (either AUROC or AUPRC). Uses all of train_batches for model 
        training, cycles through the batches in validation_batches, holding out one at a time 
        and using the others for training. Plots AUROC and/or AUPRC scores for each signal. 
        Stores trained models for each signal in IRIS object's self.models.
        
        Args:
            train_batches: list of integers of batches to always use for training
            validation_batches: list of integers of batches to cycle through for cross validation
            metric: "AUROC" or "AUPRC"
        '''
        self.anndata.obs['Clusters'] = "unknown"
        all_batches = train_batches + list(set(validation_batches) - set(train_batches))
        self.response_gene(all_batches)

        adata_full_gifford = self.anndata[np.isin(self.anndata.obs['batch'], all_batches)]

        adata_full_gifford.layers['counts'] = adata_full_gifford.X
        adata_full_gifford = adata_full_gifford.copy()

        class_names = []
        for signal in self.signals:
            class_names.append(signal + '_class')
            
        avgd_tps, avgd_precisions = None, None

        if metric == "AUROC":
            auroc_tps = []
            auroc_fps = []
            resp_tps = []
            resp_fps = []

        if metric == "AUPRC":
            auprc_precisions = []
            auprc_recalls = []
            resp_precisions = []
            resp_recalls = []

        for batch in validation_batches:
            adata_in = adata_full_gifford[~np.isin(adata_full_gifford.obs['batch'], batch)]
            adata_out = adata_full_gifford[np.isin(adata_full_gifford.obs['batch'], batch)]
            
            truth = adata_out.obs[class_names].values
            adata_out.obs[class_names] = 'unknown'
                    
            adata_full_giff2 = ad.concat([adata_in, adata_out])
            out_index = (adata_out.obs.index)

            vae = self.set_scvi_model(adata_full_giff2, epochs=25)
            
            df = pd.DataFrame({}, index=adata_full_giff2.obs.index)
            
            i = 0
            for classification in class_names:

                scanvae = scvi.model.SCANVI.from_scvi_model(vae, labels_key=classification, unlabeled_category = "unknown")
                scanvae.train(max_epochs=5, batch_size=512)
                self.models[classification] = scanvae
            
                df[classification] = scanvae.predict()
                out_results = df[np.isin(adata_full_giff2.obs.index, out_index)]
                name = classification.split('_')[0] + '_resp_zorn'

                threshold = self.find_optimal_cutoff((adata_full_gifford.obs[classification] == "Stim").values.astype(int), adata_full_gifford.obs[name])

                if metric == "AUROC":
                    tpr, fpr, _ = skm.roc_curve(truth[:, i], out_results[classification], pos_label='Stim')
                    auroc_tps.append(tpr)
                    auroc_fps.append(fpr)

                    tpr, fpr, _ = skm.roc_curve((truth[:, i] == "Stim").astype(int),  (adata_out.obs[name].values > threshold).astype(int))
                    resp_tps.append(tpr)
                    resp_fps.append(fpr)
                elif metric == "AUPRC":
                    precision, recall, _ = skm.precision_recall_curve(truth[:, i], out_results[classification], pos_label='Stim')
                    auprc_precisions.append(precision)
                    auprc_recalls.append(recall)

                    precisions, recalls, _ = skm.precision_recall_curve((truth[:, i] == "Stim").astype(int),  (adata_out.obs[name].values > threshold).astype(int))
                    resp_precisions.append(precisions)
                    resp_recalls.append(recalls)
                else:
                    print("this metric is not supported")
                    break
                    
                i += 1

        n = len(self.signals)
        m = len(validation_batches)

        # TODO: make this plotting logic smarter / less repetitive

        if metric == "AUROC":
            avgd_tps = []
            avgd_fps = []
            avgd_resp_tps = []
            avgd_resp_fps = []
            for i in range(n):
                total_tp = 0
                total_fp = 0
                for j in range(m):
                    total_tp += auroc_tps[j * n + i]
                    total_fp += auroc_fps[j * n + i]
                total_tp /= m
                total_fp /= m 
                avgd_tps.append(total_tp)
                avgd_fps.append(total_fp)

                total_tp = 0
                total_fp = 0
                for j in range(m):
                    total_tp += resp_tps[j * n + i]
                    total_fp += resp_fps[j * n + i]
                total_tp /= m
                total_fp /= m 
                avgd_resp_tps.append(total_tp)
                avgd_resp_fps.append(total_fp)

        if metric == "AUPRC":
            avgd_precisions = []
            avgd_recalls = []
            avgd_resp_precisions = []
            avgd_resp_recalls = []
            for i in range(n):
                total_precisions = 0
                total_recalls = 0
                for j in range(m):
                    total_precisions += auprc_precisions[j * n + i]
                    total_recalls += auprc_recalls[j * n + i]
                total_precisions /= m
                total_recalls /= m 
                avgd_precisions.append(total_precisions)
                avgd_recalls.append(total_recalls)

                total_precisions = 0
                total_recalls = 0
                for j in range(m):
                    total_precisions += resp_precisions[j * n + i]
                    total_recalls += resp_recalls[j * n + i]
                total_precisions /= m
                total_recalls /= m 
                avgd_resp_precisions.append(total_precisions)
                avgd_resp_recalls.append(total_recalls)
        
        colors = ['#D62728', '#17BECF', '#2CA02C', 'black', '#8C564B']
        if avgd_tps:
            f = plt.figure(1)
            for i in range(len(avgd_tps)):
                plt.plot(avgd_tps[i], avgd_fps[i], color=colors[i], dashes=(5, 5))
                plt.plot(avgd_resp_tps[i], avgd_resp_fps[i], color=colors[i], linestyle='dashed', dashes=(5, 5))
                plt.xlim([1e-6, 1])
                plt.ylim([0, 1])
            plt.legend(self.signals + ['IRIS', 'Reponse Genes'])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.title('AUROC Signaling Regressed Prediction')

        if avgd_precisions:
            g = plt.figure(2)
            for i in range(len(avgd_precisions)):
                plt.plot(avgd_recalls[i], avgd_precisions[i], color=colors[i], dashes=(5, 5))
                plt.plot(avgd_resp_recalls[i], avgd_resp_precisions[i], color=colors[i], linestyle='dashed', dashes=(5, 5))
                plt.xlim([1e-6, 1])
                plt.ylim([0, 1])
            plt.legend(self.signals + ['IRIS', 'Reponse Genes'])
            plt.ylabel('Precision')
            plt.xlabel('Recall')
            plt.title('Precision-Recall Signaling Prediction')
            
        plt.show()
    
    def calc_enrichment(
            self, 
            vae: VAE, 
            adata: AnnData
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
        '''
        Helper function that calculates enrichment score using GSEApy.

        Args:
            vae: VAE model to extract gradients from
            adata: AnnData object

        Returns:
            pre_res.res2d['NES']: 'NES' column of GSEApy preprank
            pre_res.res2d['Term']: 'Term' column of GSEApy prerank
        '''
        for param in vae.module.parameters():
            x = (param.grad)
            break
            
        arr = np.divide(np.abs(x.cpu().numpy()), 1)
        arr[(arr == np.inf)] = 0
        
        df = pd.DataFrame({'0': adata.var.index,'1':arr})
        df.index = df['0']
        df = df.drop('0', axis=1)
        
        pre_res = gp.prerank(rnk=df, # or rnk = rnk,
                        gene_sets='KEGG_2016',
                        threads=4,
                        min_size=5,
                        max_size=1000,
                        permutation_num=1000, # reduce number to speed up testing
                        outdir=None, # don't write to disk
                        seed=6,
                        verbose=True, # see what's going on behind the scenes
                        )

        # categorical scatterplot
        pre_res.res2d.index = pre_res.res2d['Term']
        
        return pre_res.res2d['NES'], pre_res.res2d['Term']

    def gsea_saliency(
            self, 
            vae: VAE, 
            batches: list[int], 
            vae_epochs: int = 20, 
            scanvae_epochs: int = 3
        ) -> pd.DataFrame:
        '''
        Performs GSEA analysis on given VAE using extracted model gradients with respect
        to each feature. Automatically plots GSEA maps for each signal and all signals
        averaged together.

        Args:
            vae: VAE model to query that SCANVI model is derived from
            batches: list of integeres of batches to use 
            vae_epochs: number of epochs to train VAE
            scanvae_epochs: number of epochs to train SCANVI from VAE

        Returns:
            df_average: pandas dataframe of calculated average scores across signals
        '''

        adata = self.anndata[np.isin(self.anndata.obs['batch'], batches)]
        dataframes = []

        for signal in self.signals: 
            df = pd.DataFrame({})
            # train vae
            for v_epoch in range(vae_epochs):
                vae.train(max_epochs=200)
                score, term = self.calc_enrichment(vae, adata)
                if df.empty:
                    df.index = term
                df['vae='+str(v_epoch)+'_scanvi=0'] = score
                scanvae = scvi.model.SCANVI.from_scvi_model(vae, labels_key=signal+'_class', unlabeled_category='unknown')
                # train SCANVI
                for sc_epoch in range(scanvae_epochs):
                    name = 'vae='+str(v_epoch)+'_scanvi=' + str(sc_epoch)
                    scanvae.train(max_epochs=9)
                    score, term = self.calc_enrichment(scanvae, adata)
                    df[name] = score
                    scanvae.save('gsea_'+signal+'_'+name)
            # save gsea scores to csv
            df.to_csv(signal + '_gsea_scanvi.csv')

        # preprocess dfs
        for signal in self.signals:
            df = pd.read_csv(signal + '_gsea_scanvi.csv')
            df.index = df['Term']
            df = df.drop('Term', axis=1)
            df = df.loc[:, df.columns.str.contains('scanvi=0')]
            dataframes.append(df)

        # calculate average scores and save
        df_average = sum(dataframes)/len(dataframes)
        df_average /= len(dataframes) 
        lst = []
        for val in df_average.index.str.split(' '):
            lst.append(' '.join(val[:-3]))
        df_average.index = lst
        df_average.columns = np.array(range(1, vae_epochs)) * 10
        df_average = df_average.sort_values(by=10, ascending=False)
        df_average.to_csv('averaged_gsea_scanvi.csv')

        j = 1
        # process and plot individual signal saliency maps
        for signal in self.signals:
            sns.set(style='white', font_scale=1, rc={'figure.figsize':(2,4)})
            plt.figure(j)
            df = pd.read_csv(signal + '_gsea_scanvi.csv')
            df.index = df['Term']
            df = df.drop('Term', axis=1)
            # remove spaces in index values
            lst = []
            for val in df.index.str.split(' '):
                lst.append(' '.join(val[:-3]))
            df.index = lst
            # df = df.loc[:, ~df.columns.str.contains('scanvi=0')] if undetermined amt of scanvi training
            df = df.loc[:, df.columns.str.contains('scanvi=1') | df.columns.str.contains('scanvi=2')]
            df.columns = [6, 9] # TODO: not sure where 6 and 9 are coming from & how to theoreotically expand
            df = df.loc[df_average.index]
            ax = sns.heatmap(df, vmin=-1.5, vmax=1.5)
            plt.savefig(signal + '_gsea_averaged_scanvi.svg')
            plt.show()
            j += 1

        # heatmap w all 5 signals
        sns.set(rc={'figure.figsize':(8,6)})
        plt.figure(j)
        ax = sns.heatmap(df_average, vmin=-1.5, vmax=1.5)
        plt.savefig('gsea_averaged.svg')
        plt.show()

        return df_average

    def gene_ablation_test(
            self, 
            modelpath: str, 
            test_batches: list[int], 
            signal: str, 
            features: list[str], 
            batch_size: int = 73,
            annot_genes: list[str] = None
        ) -> dict[str, float]:
        '''
        Takes path to SCANVI model and feature selection metrics. Randomizes batch_size number 
        of features and reruns model prediction. Currently supports mutual information 
        and highly variable genes as feature selection metrics, denoted as 'mi' and 'hv'.

        Args:
            modelpath: string of filepath to SCANVI model to be used
            test_batches: list of integers of batches to predict on
            signal: string of signal (ex. "RA")
            features: list of strings of feature selection metrics (ex. ['mi', 'hv'])
            batch_size: integer of how many genes to randomize at once
            annot_genes: list of strings of genes to annotate on plot

        Returns:
            auroc_scores: dictionary of AUROC scores keyed by feature selection metrics
        '''
        auroc_scores = {}
        adata = self.anndata
        annot_genes = set(annot_genes)

        for feature in features:
            test_results = []

            if feature == "mi": # mutual information
                output_mi = mutual_info_classif(adata.X, adata.obs[signal + '_class'])
                variable_genes = adata.var.index[output_mi.argsort()[::-1]]
            elif feature == "hv": # highly variable
                sc.pp.normalize_total(adata, target_sum=1e6, inplace=True)
                sc.pp.log1p(adata)
                sc.pp.highly_variable_genes(adata)
                adata.raw = adata
                variable_genes = adata.var['dispersions_norm'].sort_values(ascending=False).index
            else:
                print(f"{feature} selection metric is not yet supported")
                return

            df = pd.DataFrame(adata.X.todense())
            df.columns = adata.var.index
            
            i = 0
            n = len(variable_genes)
            x_axis = []
            gene_annotations = {}

            while i < n:
                if i + batch_size > n - 1:
                    genes = variable_genes[i:]
                    point = n
                else:
                    genes = variable_genes[i:i+batch_size]
                    point = i+batch_size
                x_axis.append(point)

                # randomize more genes
                goi = ''
                for val in genes:
                    df[val] = random.choices(df[val], k=len(df))
                    if val in annot_genes:
                        goi += val + ' '

                adata.X = scipy.sparse.csr_matrix(df.values)
                adata.layers['counts'] = adata.X 
                adata_out = adata.obs[signal + '_class'][np.isin(adata.obs['batch'], test_batches)]

                model = scvi.model.SCANVI.load(modelpath, adata)

                res = model.predict(adata[np.isin(adata.obs['batch'], test_batches)]) 
                score = skm.roc_auc_score(adata_out, res, pos_label = 'Stim')
                test_results.append(score)

                if goi != '':
                    gene_annotations[goi] = (point,score)

                i += batch_size
            
            # plot points
            plt.figure()
            plt.scatter(x_axis, test_results, color="gray")
            plt.xlabel(feature + ' rank')
            plt.ylabel('AUROC')

            # annotate points with genes of interest
            for label, (x_point, y_point) in gene_annotations.items():
                plt.scatter(x_point, y_point, color="red")  # highlight points
                plt.text(x_point, y_point, label, fontsize=8, color="black", ha='left', va='bottom')

            auroc_scores[feature] = test_results

        return auroc_scores
    
    def linear_score(
            self, 
            adata_in: AnnData, 
            adata_out: AnnData, 
            sigs: list[str], 
            metric: str,
            clf: Pipeline, 
            use_batch: bool = False, 
            use_celltype: bool = False, 
            use_species: bool = False
        ) -> tuple[list[float], list[float]]:
        '''
        Helper function that calculates given metric with given pipeline for each signal.

        Args:
            adata_in: AnnData object for models to be trained on
            adata_out: AnnData object for models to be tested on
            sigs: list of strings of signals
            metric: string of "AUROC", "AUPRC", or "F1"
            clf: sklearn Pipeline instance
            use_batch: boolean, train on batch identity
            use_celltype: boolean, train on celltype identity
            use_species: boolean, train on species identity

        Returns:
            lst_insample: list of metric score on training data
            lst_outsample: list of metric score on testing data
        '''
        lst_insample = []
        lst_outsample = []
        
        enc = OneHotEncoder(handle_unknown='ignore')    
        
        if ((use_batch == False) & (use_celltype == False) & (use_species == False)):
            train_x = adata_in.X.copy()
            test_x = adata_out.X.copy()
                
        elif ((use_batch == True) & (use_celltype == False) & (use_species == False)):
            batches_one_hot = enc.fit_transform(adata_in.obs['batch'].values.reshape(-1, 1))
            train_x = hstack((adata_in.X, batches_one_hot)).copy()
            batches_one_hot = enc.transform(adata_out.obs['batch'].values.reshape(-1, 1))
            test_x = hstack((adata_out.X, batches_one_hot)).copy()

        elif ((use_batch == True) & (use_celltype == False) & (use_species == True)):
            one_hot_encoding = enc.fit_transform(adata_in.obs[['batch', 'species']])
            train_x = hstack((adata_in.X, one_hot_encoding)).copy()
            one_hot_encoding = enc.transform(adata_out.obs[['batch', 'species']])
            test_x = hstack((adata_out.X, one_hot_encoding)).copy()
            
        else:
            one_hot_encoding = enc.fit_transform(adata_in.obs[['batch', 'celltype', 'species']])
            train_x = hstack((adata_in.X, one_hot_encoding)).copy()
            one_hot_encoding = enc.transform(adata_out.obs[['batch', 'celltype', 'species']])
            test_x = hstack((adata_out.X, one_hot_encoding)).copy()
            
        for val in sigs:
            train_y = adata_in.obs[val+'_class']
            test_y = adata_out.obs[val+'_class']
            clf.fit(train_x, train_y)

            in_score, out_score = None, None

            if metric == 'AUPRC':
                precision, recall, _ = skm.precision_recall_curve((train_y == "Stim").astype(int), clf.predict_proba(train_x)[:, 1])
                in_score = skm.auc(recall, precision)
                precision, recall, _ = skm.precision_recall_curve((test_y == "Stim").astype(int), clf.predict_proba(test_x)[:, 1])
                out_score = skm.auc(recall, precision)
            elif metric == 'AUROC':
                in_score = skm.roc_auc_score(train_y, clf.predict_proba(train_x)[:, 1])
                out_score = skm.roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1])
            elif metric == 'F1':
                in_score = skm.f1_score(train_y, clf.predict(train_x), pos_label = 'Stim')
                out_score = skm.f1_score(test_y, clf.predict(test_x), pos_label = 'Stim')

            lst_insample.append(in_score)
            lst_outsample.append(out_score)
        
        return lst_insample, lst_outsample
    
    def run_benchmarking(
            self, 
            adata_in: AnnData, 
            adata_out: AnnData, 
            sigs: list[str], 
            metric: str, 
            use_batch: bool = False, 
            use_celltype: bool = False, 
            use_species: bool = False
        ) -> pd.DataFrame:
        '''
        Helper function that makes evaluation pipeline and predicts using an SVM, elastic net, 
        random forest. Returns pandas dataframe of scores of linear models.

        Args:
            adata_in: AnnData object for models to be trained on
            adata_out: AnnData object for models to be tested on
            sigs: list of strings of signals
            metric: string of "AUROC", "AUPRC", or "F1"
            use_batch: boolean, train on batch identity
            use_celltype: boolean, train on celltype identity
            use_species: boolean, train on species identity

        Returns:
            df: pandas dataframe of calculated linear model scores
        '''
        df = pd.DataFrame({}, index=sigs)

        # SVM calculation
        kernel = 'linear'
        for c in np.logspace(0, 3, 3):
            in_sample, out_sample = self.linear_score(adata_in, adata_out, sigs, metric, make_pipeline(
                Normalizer(), decomposition.PCA(n_components=100, svd_solver='arpack'), svm.SVC(C=c, kernel=kernel, probability=True)
                ), use_batch, use_celltype, use_species)
            # print(kernel)
            colname = 'svm' + kernel + '_' + str(c)
            df['in-' + colname] = in_sample
            df['out-' + colname] = out_sample
                            
        # Elastic net calculation
        loss = 'log_loss'
        for penalty in ['elasticnet']:
            for alpha in np.logspace(-6, 0, 7):
                    in_sample, out_sample = self.linear_score(adata_in, adata_out, sigs, metric, make_pipeline(
                        Normalizer(), SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, max_iter=1000, tol=1e-3)
                        ), use_batch, use_celltype, use_species)
                    colname = 'sgd_' + loss + '_' + penalty + '_' + str(alpha) + '_'
                    df['in' + colname] = in_sample
                    df['out' + colname] = out_sample
    
        # Random forest calculation
        for n_estimators in [10, 100, 1000]: 
            for criterion in ['gini', 'entropy', 'log_loss']:
                for max_depth in [None, 2, 4, 8, 16]:
                    in_sample, out_sample = self.linear_score(adata_in, adata_out, sigs, metric, make_pipeline(
                        Normalizer(), decomposition.PCA(n_components=100, svd_solver='arpack'), RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth)
                        ), use_batch, use_celltype, use_species)
                    colname = 'rf_' + str(n_estimators) + '_' + criterion + '_' + str(max_depth)
                    df['in' + colname] = in_sample
                    df['out' + colname] = out_sample
        
        return df

    def linear_benchmark(
            self, 
            batches: list[int], 
            metric: str
        ) -> pd.DataFrame:
        '''
        Uses linear models (SVM, Elastic Net, Random Forest) to predict on the data, 
        calculating metric (AUROC, AUPRC, F1) for each model configuration. Provides 
        benchmark of linear model performance against IRIS performance. Holds out 
        batches one-by-one, averages metric across all batches.

        Args:
            batches: list of integers of which batches to use 
            metric: string of "AUROC", "AUPRC", or "F1"

        Returns:
            df_averaged: pandas dataframe of averaged linear model scores across batches
        '''
        adata_sub = self.anndata[np.isin(self.anndata.obs['batch'], batches)]
        filenames = []

        for batch in batches:
            in_batches = list(set(batches) - set([batch]))
            adata_in = adata_sub[np.isin(adata_sub.obs['batch'], in_batches)]
            adata_out = adata_sub[np.isin(adata_sub.obs['batch'], [batch])]
            out_name = 'benchmarking_in_'+str(in_batches)+'_out_'+str(batch)+'_'+metric+'.csv'
            score_df = self.run_benchmarking(adata_in, adata_out, self.signals, metric)
            score_df.to_csv(out_name)
            filenames.append(out_name)

        # read in scores from other files
        dataframes = []
        for filename in filenames:
            df = pd.read_csv(filename)
            df.index = df['Unnamed: 0']
            df = df.drop('Unnamed: 0', axis=1)
            dataframes.append(df)

        df_averaged = pd.concat(dataframes).groupby(by='Unnamed: 0').mean()
        df_averaged.loc[:, df.columns.str.contains('outsgd')].idxmax(axis=1)
        df_averaged.loc[:, df.columns.str.contains('out-svm')].idxmax(axis=1)
        df_averaged.loc[:, df.columns.str.contains('outrf')].idxmax(axis=1)

        # plotting
        sns.set(style='white', font_scale=1, rc={'figure.figsize':(20,8)})
        sns.heatmap(df_averaged.iloc[:, df_averaged.columns.str.contains('out')], square=True)
        plt.xticks(rotation=90)
        plt.savefig('cross-val-out-benchmarking.svg')
        
        return df_averaged
