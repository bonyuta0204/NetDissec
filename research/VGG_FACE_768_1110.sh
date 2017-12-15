#!/bin/bash
cd ..
nohup ./script/rundissect.sh --model VGG_FACE_full_conv --layers "conv5_3 fc6-conv fc7-conv fc8-conv" --dataset dataset/broden1_larger --resolution 384 --probebatch 4 --workdir dissection_test &
