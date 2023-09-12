#!/bin/bash
time=$(date "+%Y-%m-%d %H:%M:%S")
python train_Decoder.py \
  --fmri-train-file data/fmri/preprocessed_fmri/sub-03/fmri_train_data.mat \
  --img-fea-train-file data/features/ImageNetTraining/pytorch/resnet50-PCA/sub-03/feat_pca_train.mat \
  --fmri-test-file data/fmri/preprocessed_fmri/sub-03/fmri_test_data.mat \
  --img-fea-test-file data/features/ImageNetTest/pytorch/resnet50-PCA/sub-03/feat_pca_test.mat \
  --model-dir ./output/BrainDecoder \
  --CUDA \
  --gpu-id 0 \
  --batch-size 50 \
  --num-workers 4 \
  --learning-rate 8e-5 \
  --weight-decay 1e-5 \
  --max-epoch 200 \
  --snapshot-interval 10 \
  --temp1 4.0 \
  --temp2 0.2 \
  --lambda1 0.01 \
  --lambda2 0.1 \
> "./log/BrainDecoder/${time}.log"

