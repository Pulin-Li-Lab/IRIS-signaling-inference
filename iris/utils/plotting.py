# helper functions specific to plotting

import matplotlib.pyplot as plt

from numpy import arange
from pandas import DataFrame, Series
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score, roc_auc_score

def average_metrics(
        iris_x: list, 
        iris_y: list, 
        resp_x: list, 
        resp_y: list, 
        n: int, 
        m: int
    ) -> tuple[list, list, list, list]:
    '''
    Helper function that averages scores for IRIS and response gene method
    predictions. Specifically for structure of iris.cross_validate_batches.

    Args:
        iris_x: list of x-component of iris predictions (ex. recall)
        iris_y: list of y-component of iris predictions (ex. precision)
        resp_x: list of x-component of response gene method predictions
        resp_y: list of y-component of response gene method predictions
        n: integer number of signals
        m: integer number of batches averaged over

    Returns:
        avgd_iris_x:
        avgd_iris_y:
        avgd_resp_x:
        avgd_resp_y:
    '''
    avgd_iris_y = []
    avgd_iris_x = []
    avgd_resp_y = []
    avgd_resp_x = []

    for i in range(n):
        total_y = 0
        total_x = 0
        for j in range(m):
            total_y += iris_y[j * n + i]
            total_x += iris_x[j * n + i]
        total_y /= m
        total_x /= m 
        avgd_iris_y.append(total_y)
        avgd_iris_x.append(total_x)

        total_y = 0
        total_x = 0
        for j in range(m):
            total_y += resp_y[j * n + i]
            total_x += resp_x[j * n + i]
        total_y /= m
        total_x /= m 
        avgd_resp_y.append(total_y)
        avgd_resp_x.append(total_x)
    
    return avgd_iris_x, avgd_iris_y, avgd_resp_x, avgd_resp_y

def find_optimal_cutoff(
        target, 
        predicted
    ) -> list:
    '''
    Helper function to find optimal threshold of response gene value to be considered 
    positive indicator of condition. 

    Args:
        target: array-like true values 
        predicted: array-like predicted values

    Returns:
        list of threshold values
    '''
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = arange(len(tpr)) 
    roc = DataFrame({'tf' : Series(tpr-(1-fpr), index=i), 'threshold' : Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])

def plot_iris_metric(
        resp_pred_1: list,
        resp_pred_2: list,
        iris_pred_1: list,
        iris_pred_2: list,
        metric: str,
        signals: list[str],
        plot_resp: bool = False
    ) -> None:
    '''
    Plots IRIS predictions vs. response gene method predictions for a given metric. 
    Expects metric scores of predictions as inputs. Creates plot and labels/titles,
    but needs plt.show() to be called after to display both. Only AUROC/AUPRC

    Args:
        resp_pred_1: list of scores of response gene method to be plotted on x-axis 
            (ex. recall, false positive rate)
        resp_pred_2: list of scores of response gene method to be plotted on y-axis 
            (ex. precision, true positive rate)
        iris_pred_1: list of scores of IRIS predictions to be plotted on x-axis
        iris_pred_2: list of scores of IRIS predictions to be plotted on y-axis
    '''
    colors = ['#D62728', '#17BECF', '#2CA02C', 'black', '#8C564B', 'orange']

    if metric == 'F1':
        plot_f1(resp_pred_1, iris_pred_1, 'F1 Score of Predictions')

    plt.figure()
    for i in range(len(iris_pred_1)):
        if plot_resp:
            plt.plot(resp_pred_1[i], resp_pred_2[i], color=colors[i], linestyle='dashed', dashes=(5, 5))
        plt.plot(iris_pred_1[i], iris_pred_2[i], color=colors[i])
        plt.xlim([1e-6, 1])
        plt.ylim([0, 1])
    legend = signals 
    if plot_resp:
        legend += ['IRIS', 'Response Genes']
    plt.legend(legend)

    if metric == "AUROC":
        x_label = 'False Positive Rate'
        y_label = 'True Positive Rate'
        title = 'AUROC Signaling Regressed Prediction'
    elif metric == "AUPRC":
        x_label = 'Recall'
        y_label = 'Precision'
        title = 'Precision-Recall Signaling Prediction'
        
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.savefig(metric+'.png')

def score_predictions(
        iris_preds: DataFrame,
        adata,
        metric: str,
        signals: list[str],
        plot: bool = False
    ):
    '''
    Calculates score of IRIS predictions on given metric. 

    Args:
        iris_preds: pandas DataFrame of IRIS predictions, like from run_model. Expects 
            categorical predictions
        adata: AnnData object to use as ground truth of predictions
        metric: string of desired statistics; if not given, all metrics are computed
        signals: list of strings of signals to calculate score on (ex. ["Wnt"]); 
            if not given, all signals are used
        plot: whether or not to display AUROC, AUPRC curves (default True)

    Returns:
        scores: dictionary of dictionaries where keys are signals (RA, WNT, etc.) and 
            values are calculated score for that particular metric
        [IF PLOT = TRUE] --
        iris_x: list of x-component of iris predictions (ex. recall)
        iris_y: list of y-component of iris predictions (ex. precision)
        resp_x: list of x-component of response gene method predictions
        resp_y: list of y-component of response gene method predictions
    '''
    
    scores = {}
    if plot and metric != "F1":
        iris_x, iris_y, resp_x, resp_y = [], [], [], []

    for signal in signals:
        class_name = signal + '_class'
        resp_name = signal + '_resp_zorn'
        threshold = find_optimal_cutoff((adata.obs[class_name] == 'Stim').astype(int), adata.obs[resp_name])

        if metric == "AUROC":
            score = roc_auc_score((adata.obs[class_name] == 'Stim').astype(int), (iris_preds[class_name] == 'Stim').astype(int))
            if plot:
                fpr, tpr, _ = roc_curve(adata.obs[class_name], iris_preds[class_name], pos_label='Stim')
                iris_x.append(fpr)
                iris_y.append(tpr)

                fpr, tpr, _ = roc_curve((adata.obs[class_name] == "Stim").astype(int), (adata.obs[resp_name].values > threshold).astype(int))
                resp_x.append(fpr)
                resp_y.append(tpr)
        elif metric == "F1":
            score = f1_score((adata.obs[class_name] == 'Stim').astype(int), (iris_preds[class_name] == 'Stim').astype(int))
            iris_x.append(score)
            score = f1_score(adata.obs[class_name] == "Stim").astype(int), (adata.obs[resp_name].values > threshold).astype(int)
            resp_x.append(score)

        elif metric == "AUPRC":
            precision, recall, _ = precision_recall_curve(adata.obs[class_name], iris_preds[class_name], pos_label='Stim')
            score = auc(recall, precision)
            if plot:
                iris_x.append(recall)
                iris_y.append(precision)

                precisions, recalls, _ = precision_recall_curve((adata.obs[class_name] == "Stim").astype(int), (adata.obs[resp_name].values > threshold).astype(int))
                resp_x.append(recalls)
                resp_y.append(precisions)

        scores[signal] = score

    if plot:
        return scores, iris_x, iris_y, resp_x, resp_y
    else:
        return scores
    
def plot_f1(
        lst_resp: list,
        lst_nn_score: list,
        title: str,
        count: int = 1
) -> None:
    ''''
    Helper function that plots F1 scores from response gene analysis 
    vs. IRIS predictions. Calls plt.show() within the function.

    Args:
        lst_resp: response gene analysis scores
        lst_nn_score: IRIS method scores
        title: string for title of plot
        count: number of samples of scores, default 1
    '''
    plt.figure()
    plt.scatter(lst_resp, lst_nn_score, c=['#EB2027', '#29ABE2', '#00A64F', '#A87C4F', '#231F20', 'orange'] * count)
    plt.axline((0, 0), slope=1, color='k', ls='--')
    plt.xlabel('Response Gene F1 score')
    plt.ylabel('IRIS F1 score')
    plt.title(title)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()