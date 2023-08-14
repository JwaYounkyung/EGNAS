#!/bin/sh
cd "EGNAS/" 

exp_name="EGNAS_Cora"

python -m main \
    --dataset="Cora" \
    --num_individuals=20 \
    --epochs=200 \
    --shared_params \
    --half_epochs \
    --combined_evolution \
    --num_individuals_param=4 \
    --num_generations_param=5 \
    --num_parents_param=4 \
    --num_offsprings_param=2 \
    --mutate_prob=0.2 \
    --early_stopping=0 \
    --param_initial_rate=0 \
    --gpu=0 \
    2>&1 | tee ${exp_name}.log