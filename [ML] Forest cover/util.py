import matplotlib.pyplot as plt
import numpy as np
import time
import itertools


def plotClusters(X_r, y, labels, target_names, label_names):

    f, axarr = plt.subplots(1, 3)
    
    for j, c0, c1 in zip(range(3), [0,0,1], [1,2,2]):
        ax = axarr[j]
        
        for i in labels:
            target_name = target_names[i]
            ax.scatter(X_r[y == i, c0], X_r[y == i, c1], alpha=.5, lw=1,
                        label=target_name)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
    
        ax.legend(loc='best', shadow=False, scatterpoints=1)
        ax.set_title(label_names[c0] + '  vs.  ' + label_names[c1])

    
def compareClusters(X_r, y, label, target_names, label_names):
    other_labels = list(range(7))
    other_labels.remove(label)
    
    for idx in range(6):
        other_label = other_labels[idx]
        plotClusters(X_r, y, [label, other_label], target_names, label_names)

## This plots the confusion matrix
def plot_confusion_matrix(cm, target_names,title='Confusion matrix',cmap=None,normalize=False):
    
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float32') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm,2)
        

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel("Predicted label\naccuracy={:0.4f}\n misclass={:0.4f}".format(accuracy, misclass))
    plt.savefig('cm.png')
