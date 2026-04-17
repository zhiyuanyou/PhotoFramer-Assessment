import json
import numpy as np

from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr


#################### What your should modify ⬇️ ####################
# put infer output json file here
INFER_JSON = "../../src/open-r1-multimodal/result/Qwen2.5-VL-7B-GRPO-Composition-Score-Class/test_cadb_class_1k.json"
#################### What your should modify ⬆️ ####################


if __name__ == "__main__":
    with open(INFER_JSON) as fr:
        metas = json.load(fr)
    acc_sum = 0
    for meta in metas:
        preds = set([_.lower() for _ in meta["extracted_answer"]])
        gts = set([_.lower() for _ in meta["ground_truth"]])
        if preds is not None:
            num_all = max(len(preds), len(gts))
            num_right = 0
            for pred in preds:
                if pred in gts:
                    num_right += 1
                else:  # punish wrong answer
                    num_right -= 1
            acc_sum += max(num_right / num_all, 0.0)  # minimum acc for each sample is 0.0
    acc = acc_sum / len(metas)
    print(f"acc: {acc}")
