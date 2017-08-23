import numpy as np
import chainer
from chainer import cuda,  Function,  gradient_check,  report,  training,  utils,  Variable
from chainer import datasets,  iterators,  optimizers,  serializers
from chainer import Link,  Chain,  ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.links.caffe import CaffeFunction

from src import loadseg
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


MEAN=[109.5388, 118.6897, 124.6901]

def show_image(data, index):

	"""
	show image of ith image in dataset
	"""

	directory = os.path.join(data.directory, "images", data.image[index]["image"])
	image = Image.open(directory)
	image.show()
	return image

# Load data and Segmentationprefetcher
print("loading data...")
data = loadseg.SegmentationData("dataset/broden1_224")
pf = loadseg.SegmentationPrefetcher(data, categories=["image"], split=None, once=True, batch_size=1, mean=MEAN)

# prepare generator for image data in numpy array
batch = pf.tensor_batches()

# loading caffe model
print("loading caffe model...")
model = CaffeFunction("../NetDissect/zoo/caffe_reference_places365.caffemodel")
model.cleargrads()
print("caffe model loaded")

# test with first date 
for i, im in enumerate(batch):
	if i >= 3:
		break
	x = Variable(im[0])
	y = model(inputs={"data": x}, outputs=["fc8"])[0]

	x_data = x.data
	x_image = np.swapaxes(x_data, 1, 3)
	top_label=np.argmax(y.data)
	print("top label:{}".format(top_label))

	# show_image(data, i)
	plt.imshow(x_image[0])
	plt.show()


