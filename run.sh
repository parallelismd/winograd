#!/bin/bash
#SBATCH -p pac
#SBATCH -n 1 
#SBATCH -c 160
#SBATCH -o run.out
#SBATCH -e run.err

./winograd small.conf 0
./optimized small.conf 0