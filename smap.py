import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from src import loadseg
from PIL import Image

"""
module to handle saliency map data
"""


class Smap():
    def __init__(self, data, directory):
        self.directory = directory
        self.data = data
        category = os.path.join(directory, "category.dump")

        with open(category, "r") as f:
            self.category = pickle.load(f)

    def get_image_path(self, index):
        """
        get path to original image in broden
        """
        path = os.path.join(self.data.directory,
                            "images", data.image[index]["image"])
        return path

    def open_saliency_map(self, index):
        """
        get numpy array for saliency
        """
        path = os.path.join(self.directory, "{}.npy".format(index))
        print(path)
        smap = np.load(path)
        return smap

    def save_masked_image(self, index, threshold=0.8):
        """
        return original image with masking by saliency map 
        """

        smap = self.open_saliency_map(index)

        # normalize smap to range from 0 to 1
        smap = smap / smap.max()
        smap = 1 - smap
        im = Image.new("RGBA", smap.shape, "black")
        imarray = np.asarray(im)
        imarray.flags.writeable = True
        imarray = imarray.astype(np.float32)

        # changing alpha is equivalent to changing imarray[:, :, 3]
        imarray[:, :, 3] = smap
        alpha = imarray[:, :, 3]
        alpha = alpha ** 3

        alpha = alpha * 255

        condition = (np.where(alpha < 255 * threshold))

        alpha[condition] = 0
        # converting array to image and reshapre
        imarray[:, :, 3] = alpha
        im = Image.fromarray(imarray.astype(np.uint8))
        original_image = Image.open(self.get_image_path(index))
        im = im.resize(original_image.size, resample=Image.BILINEAR)

        # merge two pictures
        original_image.paste(im, (0, 0), im)

        # save image
        original_image.save(os.path.join(self.directory,
                                         "Images", "{}.png".format(index)))


if __name__ == "__main__":
    print("loading data...")
    data = loadseg.SegmentationData("dataset/broden1_227")
    smap = Smap(data, "test")
    for i in range(10):
        smap.save_masked_image(i, threshold=0.6)
