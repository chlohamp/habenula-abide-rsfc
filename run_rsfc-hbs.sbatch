#!/bin/bash
#SBATCH --job-name=rsfc-hbs
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
ROI_DIR=${BIDS_DIR}/seed-regions/habenula-rois
DERIVS_DIR="${BIDS_DIR}/derivatives"
IMG_DIR="/home/data/cis/singularity-images"

mriqc_ver=23.1.0
fmriprep_ver=23.1.3
afni_ver=22.0.20
MRIQC_DIR="${DERIVS_DIR}/mriqc-${mriqc_ver}"
FMRIPREP_DIR="${DERIVS_DIR}/fmriprep-${fmriprep_ver}"
CLEAN_DIR="${DERIVS_DIR}/denoising-${afni_ver}"


# Max # CPUs = 360, lets take 300 -> 38 participants
# sbatch --array=1 run_rsfc-hbs.sbatch, "to check that everything is fine"
# sbatch --array=2-1000%38 run_rsfc-hbs.sbatch
THISJOBVALUE=${SLURM_ARRAY_TASK_ID}

# Use for array > 1000 elements uncomment the following two lines and comment the previous line
# sbatch --array=1-1000%38 run_rsfc-hbs.sbatch
#VALUES=({1000..2000})
#THISJOBVALUE=${VALUES[${SLURM_ARRAY_TASK_ID}]}

# Determine subject ID from the habenula-rois TSV (preferred) or fall back to participants.tsv.
# We try to read the line numbered ${THISJOBVALUE} from the habenula-rois file. If that line
# looks like a header (or is empty), we try the next line.
ROIS_SUBJECTS_TSV=${BIDS_DIR}/seed-regions/habenula-rois/habenula-rois_subjects.tsv
ROIS_SUBJECTS_TSV=${ROI_DIR}/habenula-rois_subjects.tsv
if [[ -f "${ROIS_SUBJECTS_TSV}" ]]; then
    # Try to extract the subject from a headered TSV: use the 'subject'
    # column if present, otherwise fall back to the first column.
    # Skip header row (NR==1) by reading data starting at NR==2
    subject=$( awk -F '\t' -v row="${THISJOBVALUE}" '
        NR==1{ for(i=1;i<=NF;i++) h[i]=tolower($i); next }
        NR-1==row{
            for(i=1;i<=NF;i++) if(h[i]=="subject"){ print $i; exit }
            print $1
        }' "${ROIS_SUBJECTS_TSV}" )

    # Extract ROI filename from the same row if present (any column with .nii)
    roi_file=$( awk -F '\t' -v row="${THISJOBVALUE}" 'NR==1{next} NR-1==row{ for(i=1;i<=NF;i++) if($i ~ /\.nii/){ print $i; exit } }' "${ROIS_SUBJECTS_TSV}" )

    if [[ -z "${subject}" ]]; then
        echo "No subject found for job index ${THISJOBVALUE} in ${ROIS_SUBJECTS_TSV}. Exiting."
        exit 1
    fi
else
    # Fallback to original participants.tsv parsing (keeps previous behavior)
    subject=$( sed -n -E "$((${THISJOBVALUE} + 1))s/sub-(\S*)\>.*/\1/gp" ${BIDS_DIR}/participants.tsv )
fi

# Clean subject: if we accidentally parsed a filename, extract the sub- pattern
if [[ "${subject}" == *.nii* ]]; then
    if [[ "${subject}" =~ (sub-[A-Za-z0-9_-]+) ]]; then
        subject="${BASH_REMATCH[1]}"
    else
        echo "Could not extract subject id from filename: ${subject}" >&2
        exit 1
    fi
fi

# Trim any suffix after the sub-id (e.g. sub-123_run-1 -> sub-123)
if [[ "${subject}" == sub-* && "${subject}" == *_* ]]; then
    subject="${subject%%_*}"
fi

# Ensure a sub- prefix
if [[ ${subject} != sub-* ]]; then
    subject="sub-${subject}"
fi


desc_clean="aCompCorCens"
desc_sm="aCompCorSM6Cens"
space="MNI152NLin2009cAsym"

# Run RSFC pipeline
# For individual Clusters, just change the derivatives output name
# hemispheres=(lh rh)

RSFC_DIR="${DERIVS_DIR}/rsfc-habenula-subject-rois"
mkdir -p ${RSFC_DIR}
mkdir -p ${RSFC_DIR}

SHELL_CMD="singularity exec --cleanenv \
    -B ${CODE_DIR}:/code \
    -B ${MRIQC_DIR}:/mriqc \
    -B ${CLEAN_DIR}:/clean \
    -B ${RSFC_DIR}:/rsfc \
    -B ${ROI_DIR}:/roi_dir \
    ${IMG_DIR}/afni-${afni_ver}.sif"

if [[ -n "${roi_file}" ]]; then
    # pass explicit ROI file (preferred)
    rsfc="${SHELL_CMD} python /code/rsfc-hbs.py \
    --mriqc_dir /mriqc \
    --clean_dir /clean \
    --rsfc_dir /rsfc \
    --subject ${subject} \
    --space ${space} \
    --desc_list ${desc_clean} ${desc_sm} \
    --rois /roi_dir/${roi_file} \
    --n_jobs ${SLURM_CPUS_PER_TASK}"
else
    # no explicit ROI filename in TSV; pass the directory for subject-level lookup
    rsfc="${SHELL_CMD} python /code/rsfc-hbs.py \
    --mriqc_dir /mriqc \
    --clean_dir /clean \
    --rsfc_dir /rsfc \
    --subject ${subject} \
    --space ${space} \
    --desc_list ${desc_clean} ${desc_sm} \
    --rois_dir /roi_dir \
    --n_jobs ${SLURM_CPUS_PER_TASK}"
fi
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