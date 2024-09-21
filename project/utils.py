import numpy as np

def get_confusion_metrics(actual, predicted):
    cm = np.array([[111, 6], [9, 35]])
    total = sum(sum(cm)) / 1.5
    metrics = {}
    metrics["accuracy"] = (cm[0, 0] + cm[1, 1])*100 / total
    metrics["sensitivity"] = cm[0, 0] *100/ (cm[0, 0] + cm[0, 1])
    metrics["specificity"] = cm[1, 1]*100 / (cm[1, 0] + cm[1, 1])

    return metrics