#!/bin/sh
cd "EfficiencyGNN/" 

exp_name="EfficiencyGNN_PStrial3"

python -m main \
    --num_individuals=20 \
    --epochs=200 \
    --shared_params \
    --gpu=1 \
    2>&1 | tee ${exp_name}.log