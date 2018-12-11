#!/bin/bash
#SBATCH -J sort
#SBATCH -e log.err.%j
#SBATCH -o log.out.%j
#SBATCH -n 16                  # Number of procs
#SBATCH -c 1		       # 1 thread per process
#SBATCH --mem-per-cpu=1600     # Main memory in MByte for each cpu core
#SBATCH -t 00:05:00            # Hours and minutes, or '#SBATCH -t 10' - only minutes
#SBATCH --exclusive
#SBATCH --account=kurs00025
#SBATCH --partition=kurs00025
#SBATCH --reservation=kurs00025

# -------------------------------
# Afterwards you write your own commands, e.g.

echo "Start: `date`"

mpirun -n 16 ./task2

echo "End: `date`"
