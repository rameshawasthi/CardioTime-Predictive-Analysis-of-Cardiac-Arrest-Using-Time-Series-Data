
from sklearn.metrics import auc, confusion_matrix, roc_curve


def plot_history(fig, ax, history, metric="loss", suffix=""):
    ax.plot(history.history[metric], label=metric)
    ax.plot(history.history[f"val_{metric}"], label=f"validation_{metric}")
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric}: {suffix}")
    ax.legend(loc="best")
    return fig, ax



def plot_auc(fig, ax, y_test, y_pred, suffix=""):
    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    auc_keras = auc(fpr, tpr)

    ax.plot([0, 1], [0, 1], "k--")
    ax.plot(fpr, tpr, label="ROC (area = {:.3f})".format(auc_keras))
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curve: {suffix}")
    ax.legend(loc="best")

    return fig, ax


def get_confusion_metrics(actual, predicted):
    cm = confusion_matrix(actual, predicted)
    total = sum(sum(cm))
    metrics = {}
    metrics["accuracy"] = (cm[0, 0] + cm[1, 1])*100 / total
    metrics["sensitivity"] = cm[0, 0] *100/ (cm[0, 0] + cm[0, 1])
    metrics["specificity"] = cm[1, 1]*100 / (cm[1, 0] + cm[1, 1])

    return metrics
