import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from src import loadseg
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
        path = os.path.join(self.data.directory, "images", data.image[index]["image"])
        return path
    def open_saliency_map(self, index):
        """
        get numpy array for saliency
        """
        path = os.path.join(self.directory, "{}.npy".format(index))
        print(path)
        smap = np.load(path)
        return smap
    
if __name__  == "__main__":
    data = loadseg.SegmentationData("dataset/broden1_227")
    smap = Smap(data, "test")

    print(smap.category)
    print(smap.open_saliency_map(5))
