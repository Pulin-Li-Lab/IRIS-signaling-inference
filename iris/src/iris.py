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

from sklearn.feature_selection import mutual_info_classif
from csv import reader as csv_reader
from scipy.sparse import csr_matrix
from random import choices

from ..utils.generate_bitstrings import generate_bitflip, generate_random
from ..utils.plotting import average_metrics, find_optimal_cutoff, plot_iris_metric, score_predictions, plot_f1
from ..utils.benchmarking import run_benchmarking
from ..utils.analysis import get_components, calc_enrichment

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
        preset_signals = ["RA", "Bmp", "Fgf", "Wnt", "TgfB", "Shh"] 

        mappings = {}
        mappings["RA"] = ["CYP26A1", "HOXB1", "HNF1B", "CYP26C1", "HOXA1", "HOXB2", "HOXA3", "HOXA2", "HOXB3"]
        mappings["Bmp"] = ["ID2", "BMPER", "ID4", "ID1", "BAMBI", "MSX1", "ID3", "MSX2", "SMAD7"]
        mappings["Fgf"] = ["SPRY1", "SPRY2", "DUSP6", "SPRY4", "ETV5", "ETV4", "FOS", "SPRY2", "MYC", "JUNB", "DUSP14"]
        mappings["Wnt"] = ["AXIN1", "CCND1", "DKK1", "MYC", "NOTUM", "AXIN2", "SP5", "LEF1"]
        mappings["TgfB"] = ["SMAD7", "TWIST1", "TWIST2"]
        mappings["Shh"] = ["GLI1", "PTCH1", "HHIP", "FOXF1", "FOXF2"]

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
            reader = csv_reader(csvfile)
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
        adata = self.anndata[np.isin(self.anndata.obs['batch'], batches)]
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        for signal in self.signals:
            name = signal + '_resp_zorn'
            resp_lst = [gene for gene in self.pathways[signal] if gene in adata.var.index]
            mat = adata[:, resp_lst].X.todense()
            mat[np.isnan(mat)] = 0

            # assuming mat columns are in same order as resp_lst
            for i in range(len(resp_lst)):
                gene = resp_lst[i]
                weight = self.gene_weights[signal][gene]
                mat[:,i] *= weight

            std = np.std(mat, axis=1)
            std[std == 0] = 1
            mat = (mat - np.mean(mat, axis=1))/std
            lst = []
            for val in np.array(mat.sum(axis=1)):
                lst.append(val[0])
            adata.obs[name] = lst
            adata.obs[name] /= adata.obs[name].max()
            self.anndata.obs[name] = adata.obs[name]
    
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
        sc.pl.diffmap(adata_diff, components=['2, 3'], projection='2d', color='celltype')

        self.diffmaps[name] = {'data': adata_diff} 
        
    def select_diffusion_components(
            self, 
            trajectory: list[str]
        ) -> tuple[int, list[int]]:
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
        diff_components = get_components(trajectory, subsets, n_comps)
            
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
    
    def load_pretrained_model(
            self,
            paths: list[str]
        ) -> None:
        '''
        Load in pretrained models to IRIS object to be stored as models for use.
        Paths should be list of strings of paths to models. If you want to use 
        the same model for all signals, paths can be a length-1 list. If you 
        want to use different models for each signal, there should be the same 
        number of strings in paths as the number of pathways you want to predict.
        You must make sure that the original models were set up on data that is 
        compatible with the data stored in IRIS.anndata (same columns, genes, etc.).
        If you get an error from this function, you should try combining the data 
        and setting that combined data to be your IRIS.anndata.

        Args:
            paths:
        '''
        if len(paths) == 1:
            for signal in self.signals:
                self.models[signal] = scvi.model.SCANVI.load(paths[0], self.anndata)
        else:
            for i in range(len(paths)):
                self.models[self.signals[i]] = scvi.model.SCANVI.load(paths[i], self.anndata)

    
    def set_scvi_model(
            self, 
            data: AnnData, 
            n_layers: int = 2, 
            n_latent: int = 30, 
            n_hidden: int = 128, 
            epochs: int = 10,
            suffix: str = None,
            outdir: str = None
        ) -> VAE:
        '''
        Initializes SCVI model to use for IRIS analysis, returns model.

        Args:
            data: data to set up model on, passed into setup_anndata
            n_layers: number of layers model should have (default 2)
            n_latent: dimensionality of latent space (default 30)
            n_hidden: number of hidden nodes (default 128)
            epochs: maximum epochs to train SCVI model (default 10)
            suffix: string to add to end of model name (ex. test batches)
            outdir: name of directory to save models to; by default makes 
                a new directory in your folder named "models/"

        Returns:
            vae: SCVI model with given params set up on given data
        '''
        # adds fields to anndata
        scvi.model.SCVI.setup_anndata(data, layer="counts", batch_key='batch', categorical_covariate_keys=['celltype'])
        # sets up model with this anndata
        vae = scvi.model.SCVI(data, n_layers=n_layers, n_latent=n_latent, n_hidden=n_hidden, gene_likelihood="zinb")
        vae.train(max_epochs=epochs, validation_size=0.1, early_stopping=True)
        name = str(n_layers)+'_'+str(n_latent)+'_'+str(n_hidden)+'_'+str(epochs)+'_'
        if suffix:
            name += str(suffix)
        path = outdir + '/' if outdir else 'models'
        vae.save(path + name)
        return vae
        
    def run_model(
            self, 
            out_path: str, 
            outdir: str = None,
            train_batches: list[int] = None, 
            test_batches: list[int] = None, 
            n_layers: int = 2, 
            n_latent: int = 30,
            n_hidden: int = 128,
            vae_epochs: int = 400,
            scanvi_epochs: int = 5
        ) -> tuple[pd.DataFrame, AnnData]:
        '''
        Runs model with given hyperparameters and makes predictions on each signaling pathway. 
        If no VAE models stored in IRIS object, creates model with given hyperparameters, runs, 
        and stores model into IRIS object. Otherwise, loads in models stored in IRIS object and 
        runs the models to make predictions. If explicit train/test batches are given, obscures 
        celltype information from test batches and uses all of the train_batches to train. If 
        explicit batches are not given, runs with random 80/20 train/test split across all 
        batches and conditions. Stores final trained models for each signal into models attribute 
        of the IRIS object (IRIS.models). Saves model predictions dataframe into csv file at 
        outpath. Returns pandas dataframe of predictions and AnnData object representing test 
        set with predictions as a column.
        
        Args:
            out_path: string of filename to save csv of predictions to
            outdir: string of directory to save models to; by default makes a new directory 
                in your folder named "models/"
            train_batches: list of integers of batches to obscure while training 
            test_batches: list of integers of batches to keep labels for
            n_layers: number of hidden layers, default 2
            n_latent: dimensionality of latent space, default 30
            vae_epochs: integer number of epochs to train VAE, default 400
            scanvi_epochs: integer number of epochs to train SCANVI used to predict, default 5

        Returns:
            df: pandas DataFrame of prediction of presence of each signal (0 to 1)
            adata_results: AnnData of test population with IRIS predictions as columns 
        '''
        self.anndata.obs['Clusters'] = "unknown"

        # hold out random 20% of all data 
        if not train_batches and not test_batches:
            adata = self.anndata
            held_out = np.random.choice(adata.obs.index, size=int(len(adata.obs.index)*0.2))
        # hold out batches identified as test set
        else:
            adata = self.anndata[np.isin(self.anndata.obs['batch'], train_batches + test_batches)]
            held_out =  adata.obs.index[np.isin(adata.obs['batch'], test_batches)]

        adata_results = adata[held_out]
        adata.layers['counts'] = adata.X
        adata = adata.copy()
        # obscure cell type information
        if 'unknown' not in adata.obs.loc[held_out, 'celltype'].cat.categories:
            adata.obs['celltype'] = adata.obs['celltype'].cat.add_categories('unknown')
        adata.obs.loc[held_out, 'celltype'] = 'unknown'

        df = pd.DataFrame({}, index=adata_results.obs.index)

        if not self.models:
            # initialize model
            vae = self.set_scvi_model(adata, n_layers=n_layers, n_latent=n_latent, n_hidden=n_hidden, epochs=vae_epochs, outdir=outdir)

        # obscure true signal values
        class_names = []
        for signal in self.signals:
            class_names.append(signal + '_class')
            if 'unknown' not in adata.obs[signal + '_class'].cat.categories:
                adata.obs[signal + '_class'] = adata.obs[signal + '_class'].cat.add_categories('unknown')
            if sum(adata.obs[signal + '_class'].isna()) != 0:
                adata.obs[signal + '_class'].fillna("unknown", inplace=True)
            adata.obs.loc[held_out, signal + '_class'] = "unknown"

        # predict
        for val in class_names:
            if not self.models:
                scanvae = scvi.model.SCANVI.from_scvi_model(vae, labels_key=val, unlabeled_category = "unknown")
                scanvae.train(max_epochs=scanvi_epochs, check_val_every_n_epoch=1, plan_kwargs=dict(n_steps_kl_warmup=1600, n_epochs_kl_warmup=None))
                self.models[val] = scanvae
            else:
                scanvae = self.models[val]
            predictions = scanvae.predict(adata[held_out], soft=True)['Stim'].values
            df[val] = predictions
            adata_results.obs[val+'_predictions'] = predictions
            specific = val+'_vae='+str(vae_epochs)+'_scanvi='+str(scanvi_epochs)
            model_save_path = outdir+'/'+specific if outdir else 'models/'+specific
            scanvae.save(model_save_path)
        
        df.to_csv(out_path)

        return df, adata_results
    
    def find_predicted_state(
            self,
            adata: AnnData,
            state: dict[str, int]
        ) -> AnnData:
        '''
        Given particular signaling state as a dictionary ex. {'RA': 1, 'Wnt': 0, ..}, 
        identifies which cells are predicted by IRIS to have that signaling state. 
        Takes AnnData object where IRIS predictions are contained in columns (like 
        from run_model), and creates Boolean column of matches to the given state. 
        Returns given AnnData object.

        Args:
            adata: AnnData object with IRIS predictions
            state: dictionary of string of signal to 1/0 for on/off

        Returns:
            adata: AnnData object with match to state as a column
        '''
        matches = np.ones(len(adata), dtype=bool)
        # loop through each signal and check if the predictions match the mapping
        for signal, target_value in state.items():
            # map 'Stim' to 1 and 'Ctrl' to 0
            signal_matches = (adata.obs[f"{signal}_class_predictions"].map({'Stim': 1, 'Ctrl': 0}).values == target_value)
            matches_mapping &= signal_matches

        adata.obs['match'] = matches
        return adata
    
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
        if 'celltype' not in self.anndata.obs or 'batch' not in self.anndata.obs:
            print("IRIS object's AnnData must have celltype and batch columns in the obs")

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
            iris_preds: pd.DataFrame,
            batches: list[int] = None,
            signals: list[str] = None, 
            metrics: list[str] = None, 
            plot: bool = True,
            plot_resp: bool = False
        ) -> dict[str, dict[str, float]]: 
        '''
        Takes IRIS predictions and returns score of whichever statistics are asked for - 
        AUPRC, AUROC, or F1. If no pathway or metric is given, calculates all metrics 
        for all pathways. Uses stored AnnData object in IRIS.anndata as ground truth
        
        Args:
            iris_preds: pandas DataFrame of IRIS predictions. Expects categorical 
                predictions, like from run_model.
            batches: list of integers of batches from anndata object to use
            signals: list of strings of signals to calculate score on (ex. ["Wnt"]); 
                if not given, all signals are used
            metrics: list of strings of desired statistics; if not given, all metrics are computed
            plot: whether or not to display AUROC, AUPRC curves (default True)
            plot_resp: boolean to plot response gene method performance for AUROC / 
                AUPRC (default False)

        Returns:
            scores: dictionary of dictionaries where keys are signals (RA, WNT, etc.) and 
                values are dictionaries mapping metric to calculated scores
        '''
        if not signals:
            signals = self.signals
        if not metrics:
            metrics = ["AUROC", "AUPRC", "F1"]

        if batches:
            adata = self.anndata[np.isin(self.anndata.obs['batch'], batches)]
            if self.signals[0] + '_resp_zorn' not in adata.obs:
                self.response_gene(batches)
        else:
            adata = self.anndata
            batches = self.anndata.obs['batch'].unique().tolist()
            if self.signals[0] + '_resp_zorn' not in adata.obs:
                self.response_gene(batches)
            adata = adata[iris_preds['Unnamed: 0']]

        final_scores = {}

        for metric in metrics:
            scores, iris_x, iris_y, resp_x, resp_y = score_predictions(iris_preds, adata, metric, signals, plot)
            plot_iris_metric(resp_x, resp_y, iris_x, iris_y, metric, signals, plot_resp)
            for signal in signals:
                if signal not in final_scores:
                    final_scores[signal] = {}
                final_scores[signal][metric] = scores[signal]

        return final_scores 
    
    def held_out_condition_validation(
            self, 
            batches: list[int] = None, 
            condition: str = None, 
            category: str = None, 
            groupings: list[str] = None, 
            plot_each_condition: bool = False
        ) -> tuple[list[float], list[float]]:
        '''
        If no VAE models stored in IRIS object, creates model and runs. Otherwise, 
        loads in models stored in IRIS object and runs with different conditions held out.
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
        
        if batches:
            self.response_gene(batches)
            adata = self.anndata[np.isin(self.anndata.obs['batch'], batches)]
        else:
            batches = self.anndata.obs['batch'].unique().tolist()
            self.response_gene(batches)
            adata = self.anndata

        adata.layers['counts'] = adata.X
        adata = adata.copy()

        class_names = []
        for signal in self.signals:
            class_names.append(signal + '_class')
        
        # create codes for each entry
        code_lst = []
        for val in range(len(adata.obs)):
            code_lst.append(self.conv_stim_to_code(adata.obs.iloc[val]))
        adata.obs['code'] = code_lst
            
        # generate combinations to test
        if not condition:
            combos = adata.obs['code'].value_counts().index
        else:
            combos = [condition]

        lst_resp = []
        lst_nn_score = []

        if not category:
            groupings = [0]

        for grouping in groupings:
            count = 0
            for combo in combos:
                resp, nn = [], []
                count += 1
                # process data based on category, groupings, combos
                if category:
                    adata_non_cat = adata[~np.isin(adata.obs[category], grouping)]
                    adata_cat = adata[np.isin(adata.obs[category], grouping)]
                else:
                    adata_cat = adata
        
                adata_cat_out = adata_cat[np.isin(adata_cat.obs['code'], combo)]
                adata_cat_in = adata_cat[~np.isin(adata_cat.obs['code'], combo)]
                # true data values
                results = adata_cat_out.obs[class_names].values
                adata_cat_out.obs[class_names] = 'unknown'
                if category:
                    adata_full_giff2 = ad.concat([adata_non_cat, adata_cat_in, adata_cat_out])
                else:
                    adata_full_giff2 = ad.concat([adata_cat_in, adata_cat_out])
                out_index = (adata_cat_out.obs.index)
                df = pd.DataFrame({}, index=adata_full_giff2.obs.index)

                if not self.models:
                    # initialize VAE
                    vae = self.set_scvi_model(adata_full_giff2, epochs=125, suffix='held_out_'+str(count))

                i = 0
                for classification in class_names:
                    if len(results[:, i]) == 0:
                        continue
                    if not self.models:
                        # train SCANVI model
                        scanvae = scvi.model.SCANVI.from_scvi_model(vae, labels_key=classification, unlabeled_category = "unknown")
                        scanvae.train(max_epochs=5, batch_size=512)
                        self.models[classification] = scanvae
                    else:
                        scanvae = self.models[classification]
                    scanvae.save('models/'+classification+'_held_out_'+combo+'_'+str(count))
                    # make predictions
                    df[classification] = scanvae.predict()
                    out_results = df[np.isin(adata_full_giff2.obs.index, out_index)]
                
                    threshold = find_optimal_cutoff((adata_cat_in.obs[classification] == "Stim").values.astype(int), adata_cat_in.obs[classification.split('_')[0] + '_resp_zorn'])
                    
                    if (results[0, i] == 'Stim'):
                        nn.append(skm.f1_score(results[:, i], out_results[classification], pos_label='Stim'))
                        resp.append(skm.f1_score((results[:, i] == "Stim").astype(int), (adata_cat_out.obs[classification.split('_')[0] + '_resp_zorn'].values > threshold).astype(int)))
                    else:
                        nn.append(skm.f1_score(results[:, i], out_results[classification], pos_label='Ctrl'))
                        resp.append(skm.f1_score((results[:, i] == "Ctrl").astype(int), (adata_cat_out.obs[classification.split('_')[0] + '_resp_zorn'].values > threshold).astype(int)))
                    i += 1
    
                # plot each condition within each grouping
                if plot_each_condition:
                    plot_f1(resp, nn, grouping+', '+combo, legend=self.signals)
                
                lst_resp.extend(resp)
                lst_nn_score.extend(nn)

            name = 'Grouping: '+str(grouping)
            # plot per grouping, aggregating all conditions
            plot_f1(lst_resp, lst_nn_score, name, self.signals, count)
                    
        return lst_nn_score, lst_resp

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
        random_combos = list(generate_random(num_random, reference, distance))

        num_bitflip = num - num_random
        bitflip_combos = generate_bitflip(num_bitflip, [reference])

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

    def generate_screen(
            self,
            batches: list[int],
            parameters: list[list[int]],
            signal: str,
            resample_labels: bool = False,
            vae_epochs: int = 125,
            scanvi_epochs: int = 5
        ) -> None:
        '''
        Generates SCANVI models with all combinations of given parameters for all 
        stored signals. Saves models as well as csv of predictions for each signal.

        Args:
            batches: list of integers of batches to hold out one-by-one
            parameters: list of list of integers of parameter values
            signal: string of signal, ex. "RA"
            metrics: list of strings of metric, ex. ["AUROC"], ["AUROC", "F1"]
            resample_labels: boolean of whether to randomize training labels, default False
            vae_epochs: integer epochs to train VAE, default 125
            scanvi_epochs: integer epochs to train SCANVI, default 5
        '''
        hidden, layers, latent = parameters

        adata_full = self.anndata[np.isin(self.anndata.obs['batch'], batches)]
        adata_full.layers['counts'] = adata_full.X

        for batch in batches:
            adata_full = adata_full.copy()
            adata_in = adata_full[~(adata_full.obs['batch'] == batch)]
            adata_out = adata_full[adata_full.obs['batch'] == batch]

            # obscure signaling truth for held out batch
            for signal in self.signals:
                adata_out.obs[signal+'_class'] = 'unknown'
                adata_out.obs[signal+'_class'] = adata_out.obs[signal+'_class'].astype('category')
                if resample_labels:
                    adata_in.obs[signal+'_class'] = np.random.choice(['Stim', 'Ctrl'], len(adata_in))

            orig_celltypes = adata_out.obs['celltype'].cat.categories

            # obscure celltype for held out batch
            adata_out.obs['celltype'] = 'unknown'
            adata_out.obs['celltype'] = adata_out.obs['celltype'].astype('category')
            adata = ad.concat([adata_in, adata_out])

            for signal in self.signals: # make sure no Nans
                adata.obs[signal+'_class'] = adata.obs[signal+'_class'].fillna("unknown")

            # standardize celltypes
            missing = [celltype for celltype in orig_celltypes if celltype not in adata.obs['celltype'].cat.categories]
            if missing:
                adata.obs['celltype'] = adata.obs['celltype'].cat.add_categories(missing)

            in_batches = list(set(batches) - {batch})
            # suffix denotes which batches in/out
            suffix = 'in_'+str(''.join([str(elem) for elem in in_batches]))+'_out_'+str(batch)

            class_name = signal + '_class'
            outfile_name = 'vae_scanvi_layers_' + str(int(layers)) + '_hidden=' + str(int(hidden)) + '_latent=' + str(int(latent)) + '_' + class_name + '_' + suffix + '_rs.csv'
            
            df = pd.DataFrame({}, index=adata.obs.index)

            vae = self.set_scvi_model(adata, layers, latent, hidden, vae_epochs) 

            scanvae = scvi.model.SCANVI.from_scvi_model(vae, labels_key=class_name, unlabeled_category="unknown")
            scanvae.train(max_epochs=scanvi_epochs, early_stopping=True) 
            df['Stim_' + class_name + '_scanvi=' + str(scanvi_epochs) + '_vae=' + str(vae_epochs)] = scanvae.predict(soft=True)['Stim']
        
            dir_path_a = 'vae_scanvi_layers_' + str(int(layers)) + '_hidden=' + str(int(hidden)) + '_latent=' + str(int(latent)) + '_' + class_name + '_rs/'
            dir_path = 'Stim_' + class_name + '_scanvi=' + str(scanvi_epochs) + '_vae=' + str(vae_epochs)+'_'
            final_dir_path = dir_path_a + dir_path + suffix
        
            scanvae.save(final_dir_path, overwrite=True) 

            df.to_csv(outfile_name)

    def get_hyperparameter_score(
            self, 
            batches: list[int], 
            parameters: list[int], 
            signal: str, 
            metrics: list[str], 
            resample_labels: bool = False,
            vae_epochs: int = 125,
            scanvi_epochs: int = 5
        ) -> dict[str, float]:
        '''
        Creates model with given hyperparameters, makes predictions, then scores predictions 
        on the given metrics. Can also randomize the training labels before predicting on the 
        test data. Returns a dictionary of scores for each given metric.

        Args:
            batches: list of integers of batches to hold out one-by-one
            parameters: list of list of integers of parameter values
            signal: string of signal, ex. "RA"
            metrics: list of strings of metric, ex. ["AUROC"], ["AUROC", "F1"]
            resample_labels: boolean of whether to randomize training labels, default False
            vae_epochs: integer epochs to train VAE, default 125
            scanvi_epochs: integer epochs to train SCANVI, default 5
        '''
        hidden, layers, latent = parameters

        adata_full = self.anndata[np.isin(self.anndata.obs['batch'], batches)]
        adata_full.layers['counts'] = adata_full.X

        # original data 
        df_result = adata_full.obs[[signal+'_class' for signal in self.signals]]

        self.generate_screen(batches, parameters, signal, resample_labels, vae_epochs, scanvi_epochs)

        scores = {}
        score_list = []

        class_name = signal + '_class'
        fileprefix = 'vae_scanvi_layers_' + str(layers) + '_hidden=' + str(hidden) + '_latent=' + str(latent) + '_' + class_name + '_rs/'

        for metric in metrics:
            for batch in batches:
                in_batches = list(set(batches) - {batch})
                # suffix specifies which batches in/out
                suffix = 'in_'+str(''.join([str(elem) for elem in in_batches]))+'_out_'+str(batch)
                df_result_sub = df_result[adata_full.obs['batch'] == batch] # original values

                scanvae = scvi.model.SCANVI.load(fileprefix + 'Stim_' + signal + '_class_scanvi=' + str(scanvi_epochs) + '_vae=' + str(vae_epochs)+'_' + suffix, adata=adata_full)
                
                signal_lst = scanvae.predict(soft=True)['Stim'] # predictions
                signal_lst_sub = signal_lst[adata_full.obs['batch'] == batch]

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

                score_list.append(score)

            if metric not in scores:
                scores[metric] = score_list

        return scores

    def select_hyperparameters(
            self, 
            batches: list[int],
            parameters: list[list[int]], 
            signal: str, 
            metrics: list[str], 
            resample_labels: bool = False,
            vae_epochs: int = 125,
            scanvi_epochs: int = 5
        ) -> tuple[dict[str, list[int]], dict[str, NDArray[np.float64]]]:
        '''
        Samples the best combination of hyperparameters based on given parameter values. 
        Evaluates model on training data based on given metric, returns best hyperparameter 
        combination and scores for each hyperparameter combination in numpy array. 

        Args:
            batches: list of integers of batches to hold out one-by-one
            parameters: list of list of integers of parameter values
            signal: string of signal, ex. "RA"
            metrics: list of strings of metrics, ex. ["AUROC"]
            resample_labels: boolean of whether to randomize training labels, default False
            vae_epochs: integer epochs to train VAE, default 125
            scanvi_epochs: integer epochs to train SCANVI, default 5

        Returns:
            best_params: dictionary of signal to list of best parameters 
                per signal per metric
            final_scores: dictionary of metric to 

        '''
        hidden, num_layers, latent_dim = parameters
        final_scores = {metric: np.zeros((len(hidden), len(num_layers), len(latent_dim))) for metric in metrics}

        best_scores = {metric: 0 for metric in metrics}
        best_params = {metric: None for metric in metrics}

        i = 0
        for hnode in hidden: 
            j = 0
            for nlayers in num_layers:
                k = 0
                for nlat in latent_dim: 
                    params = [hnode, nlayers, nlat]
                    scores = self.get_hyperparameter_score(batches, params, signal, metrics, resample_labels, vae_epochs, scanvi_epochs)
                    for metric in metrics:
                        score = np.mean(scores[metric])
                        if score > best_scores[metric]:
                            best_scores[metric] = score
                            best_params[metric] = parameters
                        final_scores[metric][i,j,k] = score
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
            metric: str,
            vae_epochs: int = 125,
            scanvae_epochs: int = 5
        ) -> None:
        '''
        Performs cross validation holding out a list of batches and estimates given 
        performance metric (either AUROC or AUPRC). If no VAE models stored in IRIS 
        object, creates model with specified epochs, runs, and stores model back 
        into object. Otherwise, loads in models stored in IRIS object and performs 
        cross validation with the stored models. Uses all of train_batches for model 
        training, cycles through the batches in validation_batches, holding out one at a time 
        and using the others for training. Plots AUROC and/or AUPRC scores for each signal. 
        Stores trained models for each signal in IRIS object's self.models.
        
        Args:
            train_batches: list of integers of batches to always use for training
            validation_batches: list of integers of batches to cycle through for cross validation
            metric: "AUROC" or "AUPRC"
            vae_epochs: integer epochs to train VAE, default 125
            scanvi_epochs: integer epochs to train SCANVI, default 5
        '''
        self.anndata.obs['Clusters'] = "unknown"
        all_batches = train_batches + list(set(validation_batches) - set(train_batches))
        self.response_gene(all_batches)

        adata = self.anndata[np.isin(self.anndata.obs['batch'], all_batches)]

        adata.layers['counts'] = adata.X
        adata = adata.copy()

        class_names = []
        for signal in self.signals:
            class_names.append(signal + '_class')

        iris_x, iris_y, resp_x, resp_y = [], [], [], []

        for batch in validation_batches:
            adata_in = adata[~np.isin(adata.obs['batch'], batch)]
            adata_out = adata[np.isin(adata.obs['batch'], batch)]
            
            truth = adata_out.obs[class_names].values
            adata_out.obs[class_names] = 'unknown'
                    
            adata_full_giff2 = ad.concat([adata_in, adata_out])
            out_index = (adata_out.obs.index)

            if not self.models:
                vae = self.set_scvi_model(adata_full_giff2, epochs=vae_epochs)
            
            df = pd.DataFrame({}, index=adata_full_giff2.obs.index)
            
            i = 0
            for classification in class_names:
                if not self.models:
                    scanvae = scvi.model.SCANVI.from_scvi_model(vae, labels_key=classification, unlabeled_category = "unknown")
                    scanvae.train(max_epochs=scanvae_epochs, batch_size=512)
                    self.models[classification] = scanvae
                else:
                    scanvae = self.models[classification]
            
                df[classification] = scanvae.predict(soft=True)['Stim'].values
                out_results = df[np.isin(adata_full_giff2.obs.index, out_index)]
                name = classification.split('_')[0] + '_resp_zorn'

                threshold = find_optimal_cutoff((adata.obs[classification] == "Stim").values.astype(int), adata.obs[name])

                if metric == "AUROC":
                    fpr, tpr, _ = skm.roc_curve(truth[:, i], out_results[classification], pos_label='Stim')
                    iris_y.append(tpr)
                    iris_x.append(fpr)

                    fpr, tpr, _ = skm.roc_curve((truth[:, i] == "Stim").astype(int),  (adata_out.obs[name].values > threshold).astype(int))
                    resp_y.append(tpr)
                    resp_x.append(fpr)
                elif metric == "AUPRC":
                    precision, recall, _ = skm.precision_recall_curve(truth[:, i], out_results[classification], pos_label='Stim')
                    iris_y.append(precision)
                    iris_x.append(recall)

                    precisions, recalls, _ = skm.precision_recall_curve((truth[:, i] == "Stim").astype(int),  (adata_out.obs[name].values > threshold).astype(int))
                    resp_y.append(precisions)
                    resp_x.append(recalls)
                else:
                    print("this metric is not supported")
                    return
                    
                i += 1

        avgd_iris_x, avgd_iris_y, avgd_resp_x, avgd_resp_y = average_metrics(iris_x, iris_y, resp_x, resp_y)
        plot_iris_metric(avgd_resp_x, avgd_resp_y, avgd_iris_x, avgd_iris_y, metric, self.signals)
        plt.show()

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
                score, term = calc_enrichment(vae, adata)
                if df.empty:
                    df.index = term
                df['vae='+str(v_epoch)+'_scanvi=0'] = score
                scanvae = scvi.model.SCANVI.from_scvi_model(vae, labels_key=signal+'_class', unlabeled_category='unknown')
                # train SCANVI
                for sc_epoch in range(scanvae_epochs):
                    name = 'vae='+str(v_epoch)+'_scanvi=' + str(sc_epoch)
                    scanvae.train(max_epochs=9)
                    score, term = calc_enrichment(scanvae, adata)
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
            df.columns = [6, 9] 
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
        if annot_genes:
            annot_genes = set(annot_genes)
        else:
            annot_genes = set()

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
            if annot_genes:
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
                    df[val] = choices(df[val], k=len(df))
                    if val in annot_genes:
                        goi += val + ' '

                adata.X = csr_matrix(df.values)
                adata.layers['counts'] = adata.X 
                adata_out = adata.obs[signal + '_class'][np.isin(adata.obs['batch'], test_batches)]

                model = scvi.model.SCANVI.load(modelpath, adata)
                res = model.predict(adata[np.isin(adata.obs['batch'], test_batches)]) 
                score = skm.roc_auc_score(adata_out, res, pos_label = 'Stim')
                test_results.append(score)

                if goi != '':
                    gene_annotations[goi] = (point, score)

                i += batch_size
            
            # plot points
            plt.figure()
            plt.scatter(x_axis, test_results, color="gray")
            plt.xlabel(feature + ' rank')
            plt.ylabel('AUROC')

            if annot_genes:
                # annotate points with genes of interest
                for label, (x_point, y_point) in gene_annotations.items():
                    plt.scatter(x_point, y_point, color="red")  # highlight points
                    plt.text(x_point, y_point, label, fontsize=8, color="black", ha='left', va='bottom')

            auroc_scores[feature] = test_results

        return auroc_scores

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
            score_df = run_benchmarking(adata_in, adata_out, self.signals, metric)
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
        sns.heatmap(df_averaged.iloc[:, df_averaged.columns.str.contains('out')], square=True)
        plt.xticks(rotation=90)
        plt.savefig('cross-val-out-benchmarking.svg')
        
        return df_averaged
