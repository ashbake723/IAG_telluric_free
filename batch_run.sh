
#$ -V
#$ -cwd
#$ -j n
#$ -e /data3/planetgroup/ashbaker/logs/telfit.log
#$ -N telfit0
#$ -o /data3/planetgroup/ashbaker/logs/
#$ -l h_vmem=4G


echo 'here we go!'
source /home/ashbaker/miniconda2/bin/activate telfit

python /data3/planetgroup/ashbaker/IAG_Flux_Atlas/code/telfit/run.py 11 $SGE_TASK_ID


