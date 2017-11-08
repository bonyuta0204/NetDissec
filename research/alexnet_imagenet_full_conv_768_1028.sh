#!/bin/bash
cd ..
nohup ./script/rundissect.sh --model alexnet_imagenet_full_conv_768 --layers "conv1 conv2 conv3 conv4 conv5" --dataset dataset/broden1_larger/ --workdir dissection_test --resolution 384 --force pid --probebatch 16 &
