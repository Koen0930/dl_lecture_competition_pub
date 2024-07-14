#!/bin/bash

#$ -l rt_AG.small=1
#$ -l h_rt=40:00:00
#$ -N params_tuning
#$ -o /home/acf15519ro/DL_basic/dl_lecture_competition_pub/outputs/output_log.txt
#$ -e /home/acf15519ro/DL_basic/dl_lecture_competition_pub/outputs/error_log.txt
#$ -m a
#$ -m b
#$ -m e
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
source ~/DL_basic/bin/activate
module load gcc/13.2.0 python/3.10/3.10.14 cuda/11.2/11.2.2

python params.py