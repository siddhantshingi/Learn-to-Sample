#!/bin/bash
#SBATCH --job-name=LTS_cora_mean
#SBATCH -N1                          # Ensure that all cores are on one machine
#SBATCH --partition=1080ti-long             # Partition to submit to (serial_requeue)
#SBATCH --mem=4096               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=../results/LTS_norm_cora_50_run_logs_mean.out            # File to which STDOUT will be written
#SBATCH --error=../results/LTS_norm_cora_50_run_logs_mean.err            # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=30
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sshingi@umass.edu
​
echo `pwd`
# echo "SLURM task ID: "$SLURM_ARRAY_TASK_ID
#module unload cudnn/4.0
#module unload cudnn/5.1
set -x -e
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/sshingi/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/sshingi/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/sshingi/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/sshingi/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda init bash
conda activate lts
sleep 1

python3 lts.py --dataset cora --cuda 1 --meta_param_norm --model_tag LTS_norm_cora_50_mean --opt_iter 50 --feature_set v1  
​
sleep 1
exit