#!/bin/bash
cd ..
nohup ./script/rundissect.sh --model VGG_FACE_full_conv --layers "conv1_2" --dataset dataset/broden1_384 --resolution 384 --probebatch 16 &
