"""
module to compute saliency map for fully conected layers
"""

import numpy as np
import chainer
import pandas as pd
from chainer import cuda,  Function,  gradient_check,  report,  training,  utils,  Variable
from chainer import datasets,  iterators,  optimizers,  serializers
from chainer import Link,  Chain,  ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

from skimage.measure import block_reduce
import loadseg
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg

import expdir


MEAN=[109.5388, 118.6897, 124.6901]

def chainer_version_check():
    """
    raise error if chainer version is less than 2.0
    """
    ch_version = chainer.__version__
    if ch_version[0] != str(2):
        raise ImportError("chainer version 2.0 or higher is required")
   
# main function
def create_probe(
        directory, dataset, weights, mean, blobs):
	data = loadseg.SegmentationData(dataset)		
	ed = expdir.ExperimentDirectory(directory)

if __name__  == "__main__":
    chainer_version_check()
    from chainer.links.caffe import CaffeFunction
    create_probe("./dissection", "./dataset/broden1_227","./zoo/caffe_reference_places365.caffemodel", MEAN, ["fc8"] )
