#!/bin/bash
cd ..
nohup ./script/rundissect.sh --model caffe_sentibank_full_conv --layers "conv1 conv2 conv3 conv4 conv5" --dataset dataset/broden1_larger/ --workdir dissection_test/ --resolution 384 --probebatch 16 &
