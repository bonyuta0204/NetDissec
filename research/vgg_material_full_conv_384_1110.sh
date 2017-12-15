#!/bin/bash
cd ..
nohup ./script/rundissect.sh --model vgg_material_full_conv --layers "conv1_2" --dataset dataset/broden1_384  --resolution 384 --force pid --probebatch 16 &
