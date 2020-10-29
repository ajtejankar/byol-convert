#!/usr/bin/env bash

set -e
set -x

export CUDA_VISIBLE_DEVICES=$1

for pkl in *res50x1*.pkl
do
  pth="${pkl//.pkl/.pth.tar}"
	python convert.py $pkl $pth
	python validate.py $pth /datasets/imagenet/val
done

