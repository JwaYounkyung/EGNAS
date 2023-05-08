#!/bin/sh
cd "EfficiencyGNN/" 

exp_name="EfficiencyGNN_woHP_PStrial5"

python -m main \
    --dataset='CiteSeer' \
    --num_individuals=20 \
    --epochs=200 \
    --shared_params \
    --half_epochs \
    --num_individuals_param=0 \
    --num_generations_param=0 \
    --num_parents_param=0 \
    --num_offsprings_param=0 \
    --gpu=1 \
    2>&1 | tee ${exp_name}.log