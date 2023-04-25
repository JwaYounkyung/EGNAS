#!/bin/sh
cd "EfficiencyGNN/" 

exp_name="EfficiencyGNN_PStrial2"

python -m main \
    --num_individuals=20 \
    --epochs=200 \
    --mutate_prob=0.02 \
    --gpu=1 \
    2>&1 | tee ${exp_name}.log