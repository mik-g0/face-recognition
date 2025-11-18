import numpy as np
from embed import extract_folder
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def build_pairs_from_arrays(X, y):
    same_scores, diff_scores = [], []
    n = len(y)
    for i in range(n):
        for j in range(i+1, n):
            sim = float(np.dot(X[i], X[j]))  
            if y[i] == y[j]:
                same_scores.append(sim)
            else:
                diff_scores.append(sim)
    y_true = np.array([1]*len(same_scores) + [0]*len(diff_scores))
    y_score = np.array(same_scores + diff_scores)
    return y_true, y_score

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python build_pairs_and_eval.py <aligned_val_folder>")
        exit(1)
    X, y, _ = extract_folder(sys.argv[1])
    y_true, y_score = build_pairs_from_arrays(X,y)
    fpr,tpr,thr = roc_curve(y_true, y_score)
    roc_auc = auc(fpr,tpr)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = thr[eer_idx]
    print("AUC:", roc_auc, "EER:", eer, "EER_threshold:", eer_threshold)
    plt.plot(fpr,tpr,label=f"AUC={roc_auc:.4f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend()
    plt.savefig('roc.png'); print("Saved roc.png")