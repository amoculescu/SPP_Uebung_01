#!/bin/bash
#SBATCH -J QuickSortParallel
#SBATCH --mail-user=andrei.moculescu@stud.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH -e /home/kurse/kurs00025/am53teho/Praktikum1/output/Job_Name.err.%j
#SBATCH -o /home/kurse/kurs00025/am53teho/Praktikum1/output/Job_Name.out.%j
#SBATCH --mem-per-cpu=250
#SBATCH -t 00:05:00
#SBATCH -n 4
#SBATCH --account=kurs00025
#SBATCH --partition=kurs00025
#SBATCH --reservation=kurs00025
echo "This is Job $SLURM_JOB_ID"
module load gcc
module load openmpi/gcc

counter=0
while [ $counter -le 1000 ]
do
        mpirun /home/kurse/kurs00025/am53teho/Praktikum1/quickSortParallel 100000
        ((counter++))
done
