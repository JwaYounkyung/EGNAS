#!/bin/sh
cd "EfficiencyGNN/" 

exp_name="EfficiencyGNN_woHP_PStrial4_testOriented"

python -m main \
    --num_individuals=20 \
    --epochs=200 \
    --shared_params \
    --num_individuals_param=0 \
    --num_generations_param=0 \
    --num_parents_param=0 \
    --num_offsprings_param=0 \
    --gpu=1 \
    2>&1 | tee ${exp_name}.log