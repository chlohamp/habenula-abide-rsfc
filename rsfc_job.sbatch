#!/bin/bash
#SBATCH --job-name=rsfc
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4gb
#SBATCH --account=iacc_nbc
#SBATCH --qos=pq_nbc
#SBATCH --partition=IB_40C_512G
# Outputs ----------------------------------
#SBATCH --output=/home/data/nbc/Laird_ABIDE/code/log/%x/%x_%A-%a.out
#SBATCH --error=/home/data/nbc/Laird_ABIDE/code/log/%x/%x_%A-%a.err
# ------------------------------------------

set -e

pwd; hostname; date

#==============Shell script==============#
#Load the software needed
# source /home/data/abcd/code/abcd_fmriprep-analysis/env/environment
module load singularity-3.5.3

DSET_DIR="/home/data/nbc/Laird_ABIDE"
BIDS_DIR=${DSET_DIR}/dset
CODE_DIR=${DSET_DIR}/code
ROI_DIR=${BIDS_DIR}/seed-regions
DERIVS_DIR="${BIDS_DIR}/derivatives"
IMG_DIR="/home/data/cis/singularity-images"

mriqc_ver=23.1.0
fmriprep_ver=23.1.3
afni_ver=22.0.20
MRIQC_DIR="${DERIVS_DIR}/mriqc-${mriqc_ver}"
FMRIPREP_DIR="${DERIVS_DIR}/fmriprep-${fmriprep_ver}"
CLEAN_DIR="${DERIVS_DIR}/denoising-${afni_ver}"


# Max # CPUs = 360, lets take 300 -> 38 participants
# sbatch --array=1 rsfc_job.sbatch, "to check that everything is fine"
# sbatch --array=2-1000%38 rsfc_job.sbatch
# THISJOBVALUE=${SLURM_ARRAY_TASK_ID}

# Use for array > 1000 elements uncomment the following two lines and comment the previous line
# sbatch --array=1-1000%38 rsfc_job.sbatch
VALUES=({1000..2000})
# sbatch --array=1-194%38 rsfc_job.sbatch
# VALUES=({2000..2194})
THISJOBVALUE=${VALUES[${SLURM_ARRAY_TASK_ID}]}

# Parse the participants.tsv file and extract one subject ID from the line corresponding to this SLURM task.
subject=$( sed -n -E "$((${THISJOBVALUE} + 1))s/sub-(\S*)\>.*/\1/gp" ${BIDS_DIR}/participants.tsv )


desc_clean="aCompCorCens"
desc_sm="aCompCorSM6Cens"
space="MNI152NLin2009cAsym"

# Run RSFC pipeline
# For individual Clusters, just change the derivatives output name
# hemispheres=(lh rh)

PATH_TO_HABENULA_ROI="/roi_dir/hcp_hb_binavg.nii.gz"
seed_region="habenula-hcp"

clusters=(${PATH_TO_HABENULA_ROI})
RSFC_DIR="${DERIVS_DIR}/rsfc-${seed_region}-test"
mkdir -p ${RSFC_DIR}

SHELL_CMD="singularity exec --cleanenv \
    -B ${CODE_DIR}:/code \
    -B ${MRIQC_DIR}:/mriqc \
    -B ${CLEAN_DIR}:/clean \
    -B ${RSFC_DIR}:/rsfc \
    -B ${ROI_DIR}:/roi_dir \
    ${IMG_DIR}/afni-${afni_ver}.sif"

rsfc="${SHELL_CMD} python /code/rsfc.py \
    --mriqc_dir /mriqc \
    --clean_dir /clean \
    --rsfc_dir /rsfc \
    --subject sub-${subject} \
    --space ${space} \
    --desc_list ${desc_clean} ${desc_sm} \
    --rois ${clusters[@]} \
    --n_jobs ${SLURM_CPUS_PER_TASK}"
# Setup done, run the command

echo
echo Commandline: $rsfc
eval $rsfc 
exitcode=$?

# Output results to a table
echo "sub-$subject   ${THISJOBVALUE}    $exitcode" \
      >> ${CODE_DIR}/log/${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv
echo Finished tasks ${THISJOBVALUE} with exit code $exitcode

date
exit $exitcode