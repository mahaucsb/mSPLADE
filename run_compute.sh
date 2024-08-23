#!/bin/bash
#SBATCH -A csb176
#SBATCH --job-name="comp"
#SBATCH --output="comp.%j.%N.out"
#SBATCH --error="comp.%j.%N.err"
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH --gpus=1
#SBATCH -t 00:30:00

/bin/bash -c "python compute_doc_reps.py ../../../csb175/yzound/re_splade/splade_param/150000/0_MLMTransformer/ top3_docs_denoise.json"
