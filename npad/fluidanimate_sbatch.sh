#!/bin/bash

echo "Running fluidanimate sbatch"
sleep 1
sbatch run_fluidanimate_1
echo "Running fluidanimate 2 core"
sleep 1
sbatch run_fluidanimate_2
echo "Running fluidanimate 4 core"
sleep 1
sbatch run_fluidanimate_4
echo "Running fluidanimate 8 core"
sleep 1
sbatch run_fluidanimate_8
