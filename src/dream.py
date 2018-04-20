
# imports and basic notebook setup
from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
import random
import os
import argparse
import matplotlib.pyplot as plt
from IPython.display import clear_output, Image, display
from google.protobuf import text_format

import caffe

# If your GPU supports CUDA and Caffe was built with CUDA support,
# uncomment the following to run Caffe operations on the GPU.
# caffe.set_mode_gpu()
# caffe.set_device(0) # select GPU device if multiple devices exist

def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


def constuct_net(net_path, param_path):
    #net_fn   = model_path + 'alexnet_imagenet_full_conv.prototxt'
    #param_fn = model_path + 'alexnet_imagenet_full_conv.caffemodel'
    # Patching model to be able to compute gradients.
    # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_path).read(), model)
    model.force_backward = True
    open('tmp.prototxt', 'w').write(str(model))

    net = caffe.Classifier('tmp.prototxt', param_path,
                           mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                           channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB
    return net

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def objective_L2(dst):
    dst.diff[:] = dst.data 

def objective_L2_channel(dst,n):
    dst.diff[:] = 0
    dst.diff[:, n, :, :] = dst.data[:, n, :, :] 

def make_step(net, step_size=1.5, end='conv5', channel=0,
              jitter=32, clip=True, objective=objective_L2):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
            
    net.forward(end=end)
    #objective(dst)  # specify the optimization objective
    objective_L2_channel(dst, channel)
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
            
    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)    

def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, 
              end='inception_4c/output', clip=True, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
    
    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in xrange(iter_n):
            make_step(net, end=end, clip=clip, **step_params)
            
            # visualization
            # vis = deprocess(net, src.data[0])
            # if not clip: # adjust image contrast if clipping is disabled
            #     vis = vis*(255.0/np.percentile(vis, 99.98))
            # showarray(vis)
            # print octave, i, end, vis.shape
            # clear_output(wait=True)
            # 
        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])

def get_white_noise_image(width, height):
    pil_map = PIL.Image.new("RGB", (width, height), 255)
    random_grid = map(lambda x: (
            int(random.random() * 256),
            int(random.random() * 256),
            int(random.random() * 256)
        ), [0] * width * height)
    pil_map.putdata(random_grid)
    return np.asarray(pil_map)

def iter_dream(net, img, blob, n_iter=50, channel=0):
    frame = img
    frame_i = 0

    h, w = frame.shape[:2]
    s = 0.05 # scale coefficient
    for i in xrange(100):
        print("iter_{0:03d}".format(i))
        frame = deepdream(net, frame, end=blob, channel=channel)
        PIL.Image.fromarray(np.uint8(frame)).save("frames/{0}_{1:03d}_{2:04d}.jpg".format(blob, channel, frame_i))
        frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)
        frame_i += 1

class PreferedImage(object):

    def __init__(self, prototxt='/home/nakamura/network_dissection/NetDissect/zoo/alexnet_imagenet_full_conv.prototxt', caffemodel='/home/nakamura/network_dissection/NetDissect/zoo/alexnet_imagenet_full_conv.caffemodel',  root_dir= "pf_images", height=575, width=1024):
        self.net = constuct_net(prototxt, caffemodel)
        self.height = height
        self.width = width
        self.root_dir = root_dir
        self.init_image = get_white_noise_image(self.height, self.width)

    def compute_pf_image(self, blob, channel, n_iter=30):
        print("creating prefered image for {} channel {:04d}".format(blob, channel))
        frame = self.init_image
        frame_i = 0

        h, w = frame.shape[:2]
        s = 0.05 # scale coefficient
        for i in xrange(n_iter):
            print("iter_{0:03d}".format(i))
            frame = deepdream(self.net, frame, end=blob, channel=channel)
            PIL.Image.fromarray(np.uint8(frame)).save(self.save_img(blob, channel, frame_i))
            frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)
            frame_i += 1

    def save_img(self, blob, channel, n_iter):
        save_dir = os.path.join(self.root_dir, blob, "{0:04d}".format(channel))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        img_file = os.path.join(save_dir, "{0:02d}.png".format(n_iter))
        return img_file


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--blob", type=str )
    parser.add_argument("--channel", type=int)
    args = parser.parse_args()
    print(args.blob)
    print(args.channel)

    model_path = '/home/nakamura/network_dissection/NetDissect/zoo/' # substitute your path here
    prototxt = os.path.join(model_path,'alexnet_imagenet_full_conv.prototxt')
    caffemodel = os.path.join(model_path,'alexnet_imagenet_full_conv.caffemodel')

    pf = PreferedImage(prototxt, caffemodel, root_dir="../pf_images")
    pf.compute_pf_image(args.blob, args.channel)
