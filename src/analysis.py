import numpy as np
import pandas as pd

import os



class LayerStats(object):
    def __init__(self, directory, blob):
        self.blob = blob
        self.raw_iou_data = self._read_csv_summary(directory, blob)    
        self.iou_data = self._summary_data(self.raw_iou_data)    

    def _read_csv_summary(self, directory, blob):
        csv_name = os.path.join(directory, blob + "-result.csv")
        df = pd.read_csv(csv_name, index_col=0) 
        return df

    def _summary_data(self, row_data):
        summary = pd.DataFrame(index=row_data.index)
        summary["iou"]=row_data["score"]
        summary["color"] = row_data["color-iou"]
        summary["object"] = row_data["object-iou"]
        summary["material"] = row_data["material-iou"]
        summary["scene"] = row_data["scene-iou"]
        summary["part"] = row_data["part-iou"]
        summary["texture"] = row_data["texture-iou"]
        return summary

    def unit_average_stats(self):
        return self.iou_data.mean(axis=0)




if __name__  ==  "__main__":
    conv1 = LayerStats("../dissection/alexnet_imagenet_full_conv_384", "conv1")
    print(conv1.unit_average_stats())
    

