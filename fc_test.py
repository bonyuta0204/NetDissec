import numpy as np
import chainer
import pandas as pd
from chainer import cuda,  Function,  gradient_check,  report,  training,  utils,  Variable
from chainer import datasets,  iterators,  optimizers,  serializers
from chainer import Link,  Chain,  ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.links.caffe import CaffeFunction

from skimage.measure import block_reduce
from src import loadseg
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg


def show_image(data, index, ax=None):
    """
    show image of ith image in dataset
    """

    directory = os.path.join(data.directory, "images",
                             data.image[index]["image"])
    image = mpimg.imread(directory)
    if ax is None:
        ax = plt.imshow(image)
    else:
        ax.imshow(image)
    return ax

# test with first date


def saliency_map(model, batch):
    """
    generator
    return saliency map and label given by a model
    return saliency, label
    """
    i = 0
    while True:
        x = Variable(batch.next()[0])
        y = model(inputs={"data": x}, outputs=["fc8"])[0]
        model.cleargrads()

        # using top label
        top_label = np.argmax(y.data)
        top_unit = y[0, top_label]
        top_unit.backward()
        saliency = np.amax(x.grad, axis=1)[0]
        saliency = block_reduce(saliency, (4, 4), np.max)
        yield saliency, top_label


def save_saliency_map(model, batch, start=0, end=1000):
    """
    save saliency_map and classified category for images
    """
    category = {}
    for i, (smap, top_index) in enumerate(saliency_map(model, batch)):
        if start <= i < end:
            category[i] = label[top_index]
            np.save("test/{}".format(i), smap)
        elif i >= end:
            break
    with open("test/category.dump", "w") as f:
        pickle.dump(category, f)


if __name__ == "__main__":

    MEAN = [109.5388, 118.6897, 124.6901]

    label = pd.read_csv("./imalabel.csv", sep=";")
    label = label.ix[:, 1].values

    # Load data and Segmentationprefetcher
    print("loading data...")
    data = loadseg.SegmentationData("dataset/broden1_227")
    pf = loadseg.SegmentationPrefetcher(
        data, categories=["image"], split=None, once=True, batch_size=1)

    # prepare generator for image data in numpy array
    batch = pf.tensor_batches(bgr_mean=MEAN)

    # loading caffe model
    print("loading caffe model...")
    model = CaffeFunction(
        "../NetDissect/zoo/caffe_reference_imagenet.caffemodel")
    print("Caffe model loaded.")
    model.cleargrads()

    print("saving saliency_map")
    save_saliency_map(model, batch, end=1000)

    # initialize batch generator
    batch = pf.tensor_batches(bgr_mean=MEAN)
    W = 4
    H = 3
    fig, axes = plt.subplots(W, H * 2)
    axes = axes.reshape(-1)
    start = 80
    category = []
    for i, (smap, index) in enumerate(saliency_map(model, batch)):

        if start <= i < start + W * H:
            i = i - start
            axes[2 * i].imshow(smap)
            show_image(data, i + start, axes[2 * i + 1])
            axes[2 * i].axis("off")
            axes[2 * i + 1].axis("off")
            axes[2 * i].set_title(label[index])

        elif i >= W * H + start:
            break
    plt.show()
