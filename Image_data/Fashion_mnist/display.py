import matplotlib.pyplot as plt
import numpy as np

def labeled_images(images, labels, class_names, n=5):

    plt.figure(figsize=(10, 10))

    indices = np.random.randint(len(labels), size=n*n) 
    for i, val in enumerate(indices):
        plt.subplot(n, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        plt.imshow(images[val,:,:,0], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[val]])

def plot_image(idx, prob_array, y_true_array, images, class_names):
    prob, y_true, img = prob_array[idx], y_true_array[idx], images[idx]
    
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img[:,:,0], cmap=plt.cm.binary)

    y_pred = np.argmax(prob)
    if y_pred == y_true:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[y_pred],
                                          100 * np.max(prob), 
                                          class_names[y_true]), 
                                          color=color)

def plot_prob_array(idx, prob_array, y_true_array):
    prob, y_true = prob_array[idx], y_true_array[idx]

    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    fig = plt.bar(range(10), prob, color="#777777")
    plt.ylim([0, 1])
    y_pred = np.argmax(prob)

    fig[y_pred].set_color('red')
    fig[y_true].set_color('blue')

def image_prob_array(prob_array, y_true_array, test_images, class_names):

    num_rows, num_cols = 5, 3
    num_images = num_rows * num_cols

    indices = np.random.randint(len(y_true_array), size=num_images) 
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i, val in enumerate(indices):
        plt.subplot(num_rows, 2*num_cols, 2*i + 1)
        plot_image(val, prob_array, y_true_array, test_images, class_names)
        plt.subplot(num_rows, 2*num_cols, 2*i + 2)
        plot_prob_array(val, prob_array, y_true_array)


