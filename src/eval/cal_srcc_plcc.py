import json
import numpy as np

from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr


#################### What your should modify ⬇️ ####################
# put infer output json file here
INFER_JSON = "../../src/open-r1-multimodal/result/Qwen2.5-VL-7B-GRPO-Composition-Score-Class/test_cadb_score_1k.json"
#################### What your should modify ⬆️ ####################


def calculate_srcc(pred, mos):
    srcc, _ = spearmanr(pred, mos)
    return srcc


def calculate_plcc(pred, mos):
    plcc, _ = pearsonr(pred, mos)
    return plcc


def fit_curve(x, y, curve_type="logistic_4params"):
    r"""Fit the scale of predict scores to MOS scores using logistic regression suggested by VQEG.
    The function with 4 params is more commonly used.
    The 5 params function takes from DBCNN:
        - https://github.com/zwx8981/DBCNN/blob/master/dbcnn/tools/verify_performance.m
    """
    assert curve_type in [
        "logistic_4params",
        "logistic_5params",
    ], f"curve type should be in [logistic_4params, logistic_5params], but got {curve_type}."

    betas_init_4params = [np.max(y), np.min(y), np.mean(x), np.std(x) / 4.0]

    def logistic_4params(x, beta1, beta2, beta3, beta4):
        yhat = (beta1 - beta2) / (1 + np.exp(-(x - beta3) / beta4)) + beta2
        return yhat

    betas_init_5params = [10, 0, np.mean(y), 0.1, 0.1]

    def logistic_5params(x, beta1, beta2, beta3, beta4, beta5):
        logistic_part = 0.5 - 1.0 / (1 + np.exp(beta2 * (x - beta3)))
        yhat = beta1 * logistic_part + beta4 * x + beta5
        return yhat

    if curve_type == "logistic_4params":
        logistic = logistic_4params
        betas_init = betas_init_4params
    elif curve_type == "logistic_5params":
        logistic = logistic_5params
        betas_init = betas_init_5params

    betas, _ = curve_fit(logistic, x, y, p0=betas_init, maxfev=10000)
    yhat = logistic(x, *betas)
    return yhat


if __name__ == "__main__":
    with open(INFER_JSON) as fr:
        metas = json.load(fr)
    preds = []
    gts = []
    for meta in metas:
        pred = meta["extracted_answer"]
        gt = meta["ground_truth"]
        if pred is not None:
            preds.append(pred)
            gts.append(gt)
    preds = fit_curve(preds, gts)
    srcc = calculate_srcc(preds, gts)
    plcc = calculate_plcc(preds, gts)
    print(f"srcc: {srcc}, plcc: {plcc}")
