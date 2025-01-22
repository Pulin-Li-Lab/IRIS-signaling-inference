# helper functions for benchmarking with linear models

from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score, auc
from sklearn import svm, decomposition

from numpy import logspace
from pandas import DataFrame

if TYPE_CHECKING:
    from anndata import AnnData

def linear_score(
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
            precision, recall, _ = precision_recall_curve((train_y == "Stim").astype(int), clf.predict_proba(train_x)[:, 1])
            in_score = auc(recall, precision)
            precision, recall, _ = precision_recall_curve((test_y == "Stim").astype(int), clf.predict_proba(test_x)[:, 1])
            out_score = auc(recall, precision)
        elif metric == 'AUROC':
            in_score = roc_auc_score(train_y, clf.predict_proba(train_x)[:, 1])
            out_score = roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1])
        elif metric == 'F1':
            in_score = f1_score(train_y, clf.predict(train_x), pos_label = 'Stim')
            out_score = f1_score(test_y, clf.predict(test_x), pos_label = 'Stim')

        lst_insample.append(in_score)
        lst_outsample.append(out_score)
    
    return lst_insample, lst_outsample

def run_benchmarking(
        adata_in: AnnData, 
        adata_out: AnnData, 
        sigs: list[str], 
        metric: str, 
        use_batch: bool = False, 
        use_celltype: bool = False, 
        use_species: bool = False
    ) -> DataFrame:
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
    df = DataFrame({}, index=sigs)

    # SVM calculation
    kernel = 'linear'
    for c in logspace(0, 3, 3):
        in_sample, out_sample = linear_score(adata_in, adata_out, sigs, metric, make_pipeline(
            Normalizer(), decomposition.PCA(n_components=100, svd_solver='arpack'), svm.SVC(C=c, kernel=kernel, probability=True)
            ), use_batch, use_celltype, use_species)
        # print(kernel)
        colname = 'svm' + kernel + '_' + str(c)
        df['in-' + colname] = in_sample
        df['out-' + colname] = out_sample
                        
    # Elastic net calculation
    loss = 'log_loss'
    for penalty in ['elasticnet']:
        for alpha in logspace(-6, 0, 7):
                in_sample, out_sample = linear_score(adata_in, adata_out, sigs, metric, make_pipeline(
                    Normalizer(), SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, max_iter=1000, tol=1e-3)
                    ), use_batch, use_celltype, use_species)
                colname = 'sgd_' + loss + '_' + penalty + '_' + str(alpha) + '_'
                df['in' + colname] = in_sample
                df['out' + colname] = out_sample

    # Random forest calculation
    for n_estimators in [10, 100, 1000]: 
        for criterion in ['gini', 'entropy', 'log_loss']:
            for max_depth in [None, 2, 4, 8, 16]:
                in_sample, out_sample = linear_score(adata_in, adata_out, sigs, metric, make_pipeline(
                    Normalizer(), decomposition.PCA(n_components=100, svd_solver='arpack'), RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth)
                    ), use_batch, use_celltype, use_species)
                colname = 'rf_' + str(n_estimators) + '_' + criterion + '_' + str(max_depth)
                df['in' + colname] = in_sample
                df['out' + colname] = out_sample
    
    return df