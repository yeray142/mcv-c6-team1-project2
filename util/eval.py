"""
File containing main evaluation functions
"""

#Standard imports
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

#Local imports

#Constants
INFERENCE_BATCH_SIZE = 4


def evaluate(model, dataset, classes, threshold = 0.5):

    #Initialize confusion matrix
    tp = np.zeros(len(classes), np.int64)
    tn = np.zeros(len(classes), np.int64)
    fp = np.zeros(len(classes), np.int64)
    fn = np.zeros(len(classes), np.int64)

    for clip in tqdm(DataLoader(
            dataset, num_workers=4 * 2, pin_memory=True,
            batch_size=INFERENCE_BATCH_SIZE
    )):
            
        # Batched by dataloader
        batch_pred_scores = model.predict(clip['frame'])
        label = clip['label'].numpy()

        batch_pred = batch_pred_scores > threshold

        tp += (np.logical_and(label == 1, batch_pred == 1).sum(axis = 0)) #TP
        tn += (np.logical_and(label == 0, batch_pred == 0).sum(axis = 0)) #TN
        fp += (np.logical_and(label == 0, batch_pred == 1).sum(axis = 0)) #FP
        fn += (np.logical_and(label == 1, batch_pred == 0).sum(axis = 0)) #FN

    accuracies = (tp + tn) / (tp + tn + fp + fn)
    precisions = tp / (tp + fp)
    precisions[np.isnan(precisions)] = 0
    recalls = tp / (tp + fn)
    recalls[np.isnan(recalls)] = 0
    f1s = 2 * (precisions * recalls) / (precisions + recalls)
    f1s[np.isnan(f1s)] = 0

    return accuracies, precisions, recalls, f1s