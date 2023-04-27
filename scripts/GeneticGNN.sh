#!/bin/sh
cd "EfficiencyGNN/" 

exp_name="GeneticGNN"

python -m main \
    --num_individuals=20 \
    --epochs=200 \
    --num_individuals_param=4 \
    --num_generations_param=5 \
    --num_parents_param=4 \
    --num_offsprings_param=2 \
    --gpu=3 \
    2>&1 | tee ${exp_name}.log