#!/bin/bash

code=models/metrics.py
experiment_name=052_reformer_nocut_singletext_noweight_resnet50
path_pred=../drnn/experiments/${experiment_name}/test/pred_img
path_gt=../drnn/experiments/${experiment_name}/test/gt_img

gpu=0
python ${code} \
    --path_pred ${path_pred} \
    --path_gt ${path_gt} 