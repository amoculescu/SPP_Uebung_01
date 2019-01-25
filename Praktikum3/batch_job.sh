#!/bin/bash
#SBATCH -J Cuda-Lab
#SBATCH -e log.err.%j
#SBATCH -o log.out.%j
#SBATCH -n 1                  # Prozesse
#SBATCH -c 1                  # Kerne pro Prozess
#SBATCH --mem-per-cpu=1600    # Hauptspeicher in MByte pro Rechenkern
#SBATCH -t 00:02:00           # in Stunden und Minuten, oder '#SBATCH -t 10' - nur Minuten
#SBATCH --exclusive
#SBATCH -C "nvd"
#SBATCH --account=kurs00025
#SBATCH --partition=kurs00025
#SBATCH --reservation=kurs00025

# -------------------------------
# Anschliessend schreiben Sie Ihre eigenen Befehle, wie z.B.
module load cuda
./bilateral test_input.ppm 5
