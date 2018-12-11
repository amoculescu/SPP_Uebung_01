#!/bin/bash
#SBATCH -J sort
#SBATCH -e log.err.%j
#SBATCH -o log.out.%j
#SBATCH -n 48                  # Number of procs
#SBATCH -c 1		       # 1 thread per process
#SBATCH --mem-per-cpu=1600     # Main memory in MByte for each cpu core
#SBATCH --nodes=3
#SBATCH -t 00:05:00            # Hours and minutes, or '#SBATCH -t 10' - only minutes
#SBATCH --exclusive
#SBATCH --account=kurs00025
#SBATCH --partition=kurs00025
#SBATCH --reservation=kurs00025

# -------------------------------
# Afterwards you write your own commands, e.g.

echo "Start timestamp: `date`"

d=6
log_file="log"
def_iters=10
warmup_iters=1
inp_line="POINTS"

# Timing should be initialized each time before the iterations start
rm -f .cpu_freq

rm -f input.res

j=2
while [ $j -le $d ]
do
	nranks=`echo "${j} * ${j}" | bc -l`
	inp_line="${inp_line} $nranks"
	
	echo "Starting nranks ${nranks} -- `date`"
	
	i=1
	iters=$warmup_iters             # Warmup iters
	while [ $i -le $iters ]
	do
		srun -n $nranks ./task1 > /dev/null
		#srun -n $nranks --cpu-freq=HighM1-HighM1 ./task1 > /dev/null
		#mpirun -np $nranks ./task1_sol > /dev/null

		i=$(( i + 1 ))
	done

	data_line="DATA"
	i=1
	iters=$def_iters                # Normal iters
	while [ $i -le $iters ]
	do
		srun -n $nranks ./task1 > log_${nranks}_${i}
		#srun -n $nranks --cpu-freq=HighM1-HighM1 ./task1 > log_${nranks}_${i}
		#mpirun -np $nranks ./task1_sol > log_${nranks}_${i}
		exect=`cat "log_${nranks}_${i}" | grep 'Elapsed' | awk '{print $4}'`
		data_line="${data_line} $exect"
		
		i=$(( i + 1 ))
	done
	echo ${data_line} >> input.tmp
	
	rm -f log_*
	j=$(( j + 1 ))
	
	echo "Finished nranks ${nranks} -- `date`"
done

# Create the Extra-P input file
echo "X p" > input.res
echo "$inp_line" >> input.res
echo "REGION reg" >> input.res
echo "METRIC exect" >> input.res
cat input.tmp >> input.res
rm -f input.tmp

echo "End timestamp: `date`"
