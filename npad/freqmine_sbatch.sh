#!/bin/bash
## cluster, test, service
#SBATCH -p cluster
#SBATCH --job-name=FREQMINE
#SBATCH --time=0-20:00
#SBATCH --output=/home/afdafurtunato/superpc_out/freqmine-slurm-%j.out
#SBATCH --error=/home/afdafurtunato/superpc_out/freqmine-slurm-%j.err
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mail-user=alexfurtunato@gmail.com
#SBATCH --mail-type=ALL

python3 ../runparsecprocess.py -p freqmine -c gcc-hooks -i native_01:10 -r 6 1,2,6,10,14,18,22,26,30
