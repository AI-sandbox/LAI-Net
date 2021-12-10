import torch

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, jaccard_score, precision_recall_curve
from sklearn.preprocessing import label_binarize
from tabulate import tabulate


def compute_accuracy_sk(y_true, y_pred):
    correct = (y_pred == y_true).sum()
    accuracy = correct/y_true.shape[0]
    
    unique_categories = np.unique(y_true)

    accuracies = []
    for c in unique_categories:
        correct = (y_pred[y_true == c] == c).sum()
        accuracy_c = correct/np.sum(y_true == c)
        accuracies.append(accuracy_c)

    balanced_accuracy = np.mean(accuracies)
    
    return accuracy, balanced_accuracy, accuracies


def complete_sk_eval(n_classes, y_true, y_one_hot, y_pred, y_prob, cat_names=None):
    
    if cat_names is not None:
        cat_names = [str(s) for s in cat_names] #Make sure they are not ints
        
    # accuracies
    acc, acc_bal, accuracies = compute_accuracy_sk(y_true, y_pred)
    
    # sk build in report
    clf_report = classification_report(y_true, y_pred, target_names=cat_names)
    
    # Confusion Matrices
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
    
    
    # Jaccard Index
    jacc_micro = jaccard_score(y_true, y_pred, average='micro')
    jacc_macro = jaccard_score(y_true, y_pred, average='macro')
    
    # Precision/Recall Curves
    
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    mean_average_precision = 0
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_one_hot[:, i],
                                                            y_prob[:, i])
        average_precision[i] = average_precision_score(y_one_hot[:, i], y_prob[:, i])
        mean_average_precision += average_precision[i]
    mean_average_precision /= n_classes
        
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_one_hot.ravel(),
        y_prob.ravel())
    average_precision["micro"] = average_precision_score(y_one_hot, y_prob,
                                                         average="micro")
    return acc, acc_bal, accuracies, clf_report, cm, cm_norm, jacc_micro, jacc_macro, precision, recall, average_precision, mean_average_precision




def print_sk_eval(cat_names, acc, acc_bal, accuracies, clf_report, cm, cm_norm, jacc_micro, jacc_macro, precision, recall, average_precision, mean_average_precision):
    print('Hard prediction evaluation\n')
    print(clf_report)
    print('\n\n')

    print('Soft prediction evaluation\n')

    average_precision_list = [average_precision[i] for i in range(len(cat_names))]
    print(tabulate(np.hstack([np.array(cat_names)[None,:].T, np.array(average_precision_list)[None,...].T]), headers=['AP'], floatfmt=".3f"))
    print('\n\n')



    print('Confusion Matrix\n')
    print(tabulate(np.hstack([np.array(cat_names)[None,:].T, cm]), headers=cat_names))
    print('\n\n')
    print('Confusion Matrix (Normalized)\n')
    print(tabulate(np.hstack([np.array(cat_names)[None,:].T, cm_norm*100]), headers=cat_names, floatfmt=".1f"))

    print('\n\n')
    print('Summary: \n')
    print('Accuracy {}'.format(acc))
    print('Balanced Accuracy {}'.format(acc_bal))
    print('Jaccard Index (micro) {}'.format(jacc_micro))
    print('Jaccard Index (macro) {}'.format(jacc_macro))
    print('Mean (macro) Average Precision {}'.format(mean_average_precision))



def compute_accuracy(pred, labels, balanced_accuracy = True, return_accuracies_list=False):
    if balanced_accuracy == False:
        correct = (pred == labels).float().sum()
        accuracy = correct/labels.shape[0]
    else:
        unique_categories = torch.unique(labels)

        accuracies = []
        for c in unique_categories:
            correct = (pred[labels == c] == c).float().sum()
            accuracy_c = correct/torch.sum(labels == c)
            accuracies.append(accuracy_c.cpu().item())

        accuracy = torch.mean(torch.tensor(accuracies))
    
    if return_accuracies_list:
        return accuracy, accuracies
    return accuracy



class AccuracyLogger:
    def __init__(self, name):
        self.name = name
        self.current_accuracy = 0.0
        self.best_accuracy = 0.0
        self.time_since_best = 0
        
    def __str__(self):
        return '{} accuracy is: {:.2f} - Best is: {:.2f} - Difference is: {:.2f} - Time since best: {}'.format(self.name, self.current_accuracy*100, self.best_accuracy*100, 100*(self.best_accuracy-self.current_accuracy), self.time_since_best)
        
    def log(self, accuracy):
        is_best = False
        self.current_accuracy = accuracy
        if self.current_accuracy > self.best_accuracy:
            self.best_accuracy = self.current_accuracy
            self.time_since_best = 0
            is_best = True
        else:
            self.time_since_best += 1
        return is_best
            
    def log_and_print(self, accuracy):
        is_best = self.log(accuracy)
        print(self)
        return is_best