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
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg

MEAN=[109.5388, 118.6897, 124.6901]

label = pd.read_csv("./imalabel.csv", sep=";")
label = label.ix[:, 1].values

def show_image(data, index, ax=None):
	"""
	show image of ith image in dataset
	"""

	directory = os.path.join(data.directory, "images", data.image[index]["image"])
	image = mpimg.imread(directory) 
	if ax is None:
		ax = plt.imshow(image) 
	else:
		ax.imshow(image)
	return ax 

# Load data and Segmentationprefetcher
print("loading data...")
data = loadseg.SegmentationData("dataset/broden1_227")
pf = loadseg.SegmentationPrefetcher(data, categories=["image"], split=None, once=True, batch_size=1) 

# prepare generator for image data in numpy array
batch = pf.tensor_batches(bgr_mean=MEAN)

# loading caffe model
print("loading caffe model...")
model = CaffeFunction("../NetDissect/zoo/caffe_reference_imagenet.caffemodel")
model.cleargrads()
print("caffe model loaded")

# test with first date 
for i, im in enumerate(batch):
	if i >= 8:
		break
	x = Variable(im[0])
	y = model(inputs={"data": x}, outputs=["fc8"])[0]
	model.cleargrads()
	x_data = x.data

	# using top label
	top_label=np.argmax(y.data)
	print("top label:{}".format(top_label))
	top_unit = y[0, top_label]
	top_unit.backward()
	saliency = np.amax(x.grad, axis=1)[0]
	saliency = block_reduce(saliency, (2, 2), np.max)

	# plot image and saliency map
	plt.figure()
	ax1 = plt.subplot(1, 2, 1)
	ax2 = plt.subplot(1, 2, 2)
	show_image(data, i, ax=ax1)
	im = ax2.imshow(saliency)
	plt.colorbar(im)
	plt.title(label[top_label])	
plt.show()



