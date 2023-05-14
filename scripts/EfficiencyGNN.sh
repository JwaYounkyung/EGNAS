#!/bin/sh
cd "EfficiencyGNN/" 

exp_name="EfficiencyGNN_CEtrial1_mutation0.2"

python -m main \
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
    --gpu=1 \
    2>&1 | tee ${exp_name}.log