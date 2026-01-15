#!/bin/bash
#SBATCH --job-name=age-conn
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4gb
#SBATCH --account=iacc_nbc
#SBATCH --qos=pq_nbc
#SBATCH --partition=IB_16C_96G
# Outputs ----------------------------------
#SBATCH --output=/home/data/nbc/Laird_ABIDE/code/log/rsfc-age/%x_%A-%a.out
#SBATCH --error=/home/data/nbc/Laird_ABIDE/code/log/rsfc-age/%x_%A-%a.err
# ------------------------------------------

set -e

pwd; hostname; date

# IB_16C_96G, IB_40C_512G, IB_44C_512G
# Extract age and habenula connectivity for each binary ROI from age-effects analysis
# sbatch --array=0-N age-conn-hbs_job.sbatch (where N = number of clusters - 1)
# sbatch --array=0 age-conn-hbs_job.sbatch (where N = number of clusters - 1)

#==============Shell script==============#
#Load the software needed
# source /home/data/abcd/code/abcd_fmriprep-analysis/env/environment
module load singularity-3.8.2

mriqc_ver=23.1.0
fmriprep_ver=23.1.3
afni_ver=22.3.05 # Use this version to run lmer with missing data

DSET_DIR="/home/data/nbc/Laird_ABIDE"
BIDS_DIR=${DSET_DIR}/dset
CODE_DIR=${DSET_DIR}/code
DERIVS_DIR="${BIDS_DIR}/derivatives"
IMG_DIR="/home/data/cis/singularity-images"

MRIQC_DIR="${DERIVS_DIR}/mriqc-${mriqc_ver}"

TEMPLATE_DIR="/home/data/cis/templateflow/tpl-MNI152NLin2009cAsym"
template="tpl-MNI152NLin2009cAsym_res-02_desc-brain_T1w.nii.gz"
template_mask="tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz"


seed_region="habenula"
RSFC_DIR="${DERIVS_DIR}/rsfc-${seed_region}-subject-rois"
GROUP_DIR="${RSFC_DIR}/group" 

ROIs=("habenula")
ROI=${ROIs[0]}

# Calculate cluster ID (array task ID + 1)
CLUSTER_ID=$((SLURM_ARRAY_TASK_ID + 1))


# Age-effects cluster ROI directory
AGE_DIR="${RSFC_DIR}/age-effect5-21"

# Bind directories to singularity image
SHELL_CMD="singularity exec --cleanenv \
    -B ${BIDS_DIR}:/data \
    -B ${CODE_DIR}:/code \
    -B ${GROUP_DIR}:/group \
    -B ${TEMPLATE_DIR}:/template_dir \
    -B ${RSFC_DIR}:/rsfc \
    -B ${AGE_DIR}:/age \
    ${IMG_DIR}/afni-${afni_ver}.sif"

# Pass inputs
analysis="${SHELL_CMD} python /code/age-conn-hbs.py \
    --dset /data \
    --group_dir /group \
    --rsfc_dir /rsfc \
    --template /template_dir/${template} \
    --template_mask /template_dir/${template_mask} \
    --roi_lst ${ROIs[@]} \
    --roi ${ROI} \
    --age_dir /age \
    --cluster_id ${CLUSTER_ID} \
    --n_jobs ${SLURM_CPUS_PER_TASK}"


# Setup done, run the command
echo
echo Commandline: $analysis
eval $analysis 
exitcode=$?

echo Finished tasks with exit code $exitcode
date

exit $exitcode
