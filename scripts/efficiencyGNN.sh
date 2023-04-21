#!/bin/sh
cd "EfficiencyGNN/" 

exp_name="efficiencyGNN_trial1"

python -m main \
    --num_individuals=20 \
    --epochs=200 \
    2>&1 | tee ${exp_name}.log