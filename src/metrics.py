import numpy as np
from sklearn.metrics import confusion_matrix, fbeta_score


def business_cost(y_true, y_pred, fn_cost=10, fp_cost=1):

    cm = confusion_matrix(y_true, y_pred)
    # Structure de cm : [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()

    total_cost = (fn * fn_cost) + (fp * fp_cost)

    # On normalise le coût par le nombre de clients pour avoir une métrique comparable
    normalized_cost = total_cost / len(y_true)

    return normalized_cost


def find_optimal_threshold(y_true, y_proba, fn_cost=10, fp_cost=1):
    #meilleur seuil, meilleur cout
    thresholds = np.arange(0.0, 1.01, 0.01)
    costs = []

    for thresh in thresholds:
        # Si proba > seuil => 1 (Refus du prêt car risque de défaut)
        y_pred = (y_proba >= thresh).astype(int)
        cost = business_cost(y_true, y_pred, fn_cost, fp_cost)
        costs.append(cost)

    best_index = np.argmin(costs)
    return thresholds[best_index], costs[best_index]