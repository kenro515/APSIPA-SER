import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_cm(cm_target, cm_predict, x_label=None, y_label=None, dir_path_name=None):
    time_plot_now = datetime.datetime.now()
    cm = confusion_matrix(cm_target, cm_predict)
    cm = cm.astype('float64') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10) 

    class_label = ['ang', 'joy', 'sad', 'neu']
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_label)),
        yticks=np.arange(len(class_label)),
        xticklabels=class_label,
        yticklabels=class_label,
        xlabel=x_label,
        ylabel=y_label
    )
    data_format = '.2f'

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], data_format),
                ha='center',
                va='center'
            )
    fig.tight_layout()
    # -----------------------------------

    plt.savefig(
        './results/{}/confusion_matrix/{}.eps'.format(
            dir_path_name, time_plot_now)
    )


def plot_curve(p_train_loss, p_valid_loss, x_label=None, y_label=None, title=None, fold_idx=None, dir_path_name=None):
    plt.clf()
    plt.plot(p_train_loss, linewidth=1, label="train")
    plt.plot(p_valid_loss, linewidth=1, label="valid")
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend()
    plt.grid()
    if title == 'Learning curve':
        plt.savefig(
            './results/{}/learning_curve/fold_{}.eps'.format(
                dir_path_name, fold_idx),
            bbox_inches="tight",
            pad_inches=0.1
        )
    else:
        plt.savefig(
            './results/{}/accuracy_curve/fold_{}.eps'.format(
                dir_path_name, fold_idx),
            bbox_inches="tight",
            pad_inches=0.1
        )
