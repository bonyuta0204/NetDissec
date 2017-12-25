import sys
import os

sys.path.append("/home/nakamura/network_dissection/NetDissect/src")

from loadseg import SegmentationData

class category_info(object):
    def __init__(self, dataset = "/home/nakamura/network_dissection/NetDissect/dataset/broden1_384/"):
        self.ds = SegmentationData(dataset)
        
        # initialize label_to_index
        self.label_to_index = {}
        i = 0
        while True:
            try:
                self.label_to_index[self.ds.name(None, i)] = i
                i += 1
            except IndexError:
                break
        self.index_to_category = self.ds.primary_categories_per_index()
    def is_texture(self,label):
        """check if label is texture"""
        try:
            index = self.label_to_index[label]
            return (self.index_to_category[index]) == 5
        except:
            return False
    
    def generate_color_list(self, label_list, texture_c="red", no_texture_c="blue"):
        """
        generate color list for plotting.
        """
        colors = []
        for label in label_list:
            if self.is_texture(label):
                colors.append(texture_c)
            else:
                colors.append(no_texture_c)
        return colors
