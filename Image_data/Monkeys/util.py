import matplotlib.pyplot as plt
import numpy as np
import time
import itertools


def display_image(fname):
    
    img = plt.imread(fname)
    print(img.shape)

    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img)

def extract_features(sample_count, batch_size, generator, base_model):
    
    start = time.time()
    features =  np.zeros(shape=(sample_count, 10, 10, 2048))
    labels = np.zeros(shape=(sample_count,10))
    
    i = 0
    for inputs_batch, labels_batch in generator:
        stop = time.time()
        dt = stop - start
        print('\r',
              'Extracting features from batch', str(i+1), '/', len(generator),
              '-- run time:', dt,'seconds',
              end='')
        
        features_batch = base_model.predict(inputs_batch)
        
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        
        if i * batch_size >= sample_count:
            break
            
    print("\n")
    
    return features, labels


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

