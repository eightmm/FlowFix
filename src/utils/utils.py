import torch, numpy, random

from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix, mean_absolute_error, mean_squared_error , r2_score
from scipy.stats import kendalltau, spearmanr

def format_pytorch_version(version): return version.split('+')[0]

def format_cuda_version(version): return 'cu' + version.replace('.', '')

def format_pyg_version(version): return version

def format_dgl_version(version): return version

def set_random_seed(seed=0):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'RANDOM, DGL, NUMPY and TORCH random seed is set {seed}.')

def train_valid_test(data, num_valid, num_test):
    total_range = range( len(data) )
    valid_indices = random.sample( total_range, num_valid )
    unused_indices = list( set(total_range) - set(valid_indices) )
    test_indices = random.sample( unused_indices, num_test )
    train_indices = list( set(total_range) - set(valid_indices) - set(test_indices) )
    return train_indices, valid_indices, test_indices

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params:,}")

def get_save_path( hyperparameter ):
    hyp = ""
    for key, value in hyperparameter.items():
        words = key.split('_')
        check = ""
        for word in words:
            check += word[0]
        hyp += f'{check}{value}_'
    return hyp

def analysis_regression(pred, true):
    mae = mean_absolute_error( true, pred )
    mse = mean_squared_error ( true, pred )
    rmse = mse ** 0.5
    kendall  = kendalltau(true, pred)[0]
    spearman = spearmanr(true, pred)[0]
    corrcoef = numpy.corrcoef(true, pred)[0][1]
    r_squared = r2_score(true, pred)

    result = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "Kendall Tau": kendall,
        "Spearman r": spearman,
        "Pearson r": corrcoef,
        "R^2": r_squared,
    }
    return result

# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error, explained_variance_score, max_error, median_absolute_error
# from scipy.stats import kendalltau, spearmanr
# import numpy

# def analysis_regression(pred, true):
#     mae = mean_absolute_error(true, pred)
#     mse = mean_squared_error(true, pred)
#     rmse = mse ** 0.5
#     msle = mean_squared_log_error(true, pred)
#     rmsle = msle ** 0.5
#     kendall = kendalltau(true, pred)[0]
#     spearman = spearmanr(true, pred)[0]
#     corrcoef = numpy.corrcoef(true, pred)[0][1]
#     r_squared = r2_score(true, pred)
#     explained_var = explained_variance_score(true, pred)
#     max_err = max_error(true, pred)
#     median_ae = median_absolute_error(true, pred)
#     mape = numpy.mean(numpy.abs((true - pred) / true)) * 100

#     result = {
#         "MAE": mae,
#         "MSE": mse,
#         "RMSE": rmse,
#         "MSLE": msle,
#         "RMSLE": rmsle,
#         "Kendall Tau": kendall,
#         "Spearman R": spearman,
#         "Pearson R": corrcoef,
#         "R^2": r_squared,
#         "Explained Variance": explained_var,
#         "Max Error": max_err,
#         "Median AE": median_ae,
#         "MAPE (%)": mape,
#     }
# #     return result



def cal_mcc(sensitivity, specificity, prevalence):
    assert sensitivity >= 0, "Sensitivity have to bigger than 0"
    assert specificity >= 0, "specificity have to bigger than 0"

    numerator = sensitivity + specificity - 1
    denominatorFirstTerm = sensitivity + (1 - specificity)*(1 - prevalence) / prevalence
    denominatorSecondTerm = specificity + (1 -sensitivity)*prevalence/(1 - prevalence) 
    denominator = (denominatorFirstTerm * denominatorSecondTerm) ** 0.5

    if sensitivity == 1 and specificity == 0:
        denominator = 1
    if sensitivity == 0 and specificity == 1:
        denominator = 1.

    return (numerator / denominator)

def analysis_binary(pred, true, grid=0.5, prevalence=0.5):
    try:
        roc_auc    = roc_auc_score(true, pred)
    except Exception as E:
        roc_auc    = 0

    true = [ 1 if i > grid else 0 for i in true ]
    pred = [ 1 if i > grid else 0 for i in pred ]

    binary_acc     = accuracy_score(true, pred)
    precision      = precision_score(true, pred)
    f1             = f1_score(true, pred)

    mcc            = matthews_corrcoef(true, pred)
    TN, FP, FN, TP = confusion_matrix(true, pred).ravel()
    sensitivity    = 1.0 * TP / (TP + FN)
    specificity    = 1.0 * TN / (FP + TN)
    NPV            = 1.0 * TN / (TN + FN)
    bal_mcc = cal_mcc(sensitivity, specificity, prevalence=prevalence)

    result = {
        'ACC': binary_acc,
        'MCC': mcc,
        'Bal_MCC': bal_mcc,
        'Sensitivity Recall': sensitivity,
        'Specificity': specificity,
        'Precision PPV': precision,
        'NPV': NPV,
        'F1': f1,
        'ROC_AUC': roc_auc,
    }
    return result

def analysis_table(dct, type='float'):
    table = PrettyTable()
    for c in dct.keys():
        table.add_column(c, [])
    if type == 'int':
        table.add_row([ f'{int(c)}' for c in dct.values()])
    else:
        table.add_row([ f'{float(c):.3f}' for c in dct.values()])
    print(table, flush=True)
    return table