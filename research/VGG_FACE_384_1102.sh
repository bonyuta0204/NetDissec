#!/bin/bash
cd ..
./script/rundissect.sh --model VGG_FACE_full_conv --layers "conv5_3 fc6-conv fc7-conv fc8-conv" --dataset dataset/broden1_384 --resolution 384 --probebatch 16
