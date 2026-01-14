import argparse
import os
import os.path as op
import string
from glob import glob

import nibabel as nib
import numpy as np
from nilearn._utils.niimg_conversions import _check_same_fov
import pandas as pd
from nilearn import image, masking

from nilearn.maskers import NiftiMasker

def _get_parser():
    parser = argparse.ArgumentParser(description="Run RSFC in AFNI")
    parser.add_argument(
        "--dset",
        dest="dset",
        required=True,
        help="Path to BIDS directory",
    )
    parser.add_argument(
        "--group_dir",
        dest="group_dir",
        required=True,
        help="Path to group directory",
    )
    parser.add_argument(
        "--rsfc_dir",
        dest="rsfc_dir",
        required=True,
        help="Path to RSFC directory",
    )
    parser.add_argument(
        "--template",
        dest="template",
        default=None,
        required=False,
        help="Template to resample data",
    )
    parser.add_argument(
        "--template_mask",
        dest="template_mask",
        default=None,
        required=False,
        help="Template to resample masks",
    )
    parser.add_argument(
        "--roi_lst",
        dest="roi_lst",
        nargs="+",
        required=True,
        help="ROI label list",
    )
    parser.add_argument(
        "--roi",
        dest="roi",
        required=True,
        help="ROI label",
    )
    parser.add_argument(
        "--pheno_dir",
        dest="pheno_dir",
        required=True,
        help="Path to directory containing cluster ROI masks",
    )
    parser.add_argument(
        "--cluster_id",
        dest="cluster_id",
        required=True,
        help="Cluster ID to process (1-4)",
    )
    parser.add_argument(
        "--n_jobs",
        dest="n_jobs",
        default=4,
        required=False,
        help="CPUs",
    )
    return parser

def afni2nifti(afni_fn, nifti_fn):
    cmd = f"3dAFNItoNIFTI \
                -prefix {nifti_fn} \
                {afni_fn}"
    print(f"\t\t\t{cmd}", flush=True)
    os.system(cmd)

def main(
    dset,
    group_dir,
    rsfc_dir,
    template,
    template_mask,
    roi_lst,
    roi,
    pheno_dir,
    cluster_id,
    n_jobs,
):
    """Run group analysis workflows on a given dataset."""
    os.system(f"export OMP_NUM_THREADS={n_jobs}")
    roi_dict = {label: x * 3 + 1 for x, label in enumerate(roi_lst)}
    print(roi_dict, flush=True)
    space = "MNI152NLin2009cAsym"
    n_jobs = int(n_jobs)

    # Load the specified cluster ROI mask from pheno directory
    roi_file = op.join(pheno_dir, f"cluster_{cluster_id.zfill(2)}_roi.nii.gz")
    if not op.exists(roi_file):
        raise FileNotFoundError(f"Cluster {cluster_id} ROI not found at {roi_file}")
    
    print(f"Processing cluster {cluster_id} ROI: {roi_file}", flush=True)

    group_path = os.path.join(group_dir, roi)
    phenotypic_df = pd.read_csv(op.join(dset, "participants.tsv"), sep="\t")
    #phenotypes = ["ADOS_GOTHAM_SOCAFFECT", "ADOS_GOTHAM_RRB", "SRS_MOTIVATION", "VINELAND_DAILYLVNG_STANDARD", "VINELAND_COPING_V_SCALED", ]
    #phenotypes = ["ADI_RRB_TOTAL_C", "ADI_R_RRB_TOTAL_C", "ADI_R_SOCIAL_TOTAL_A", "ADI_R_VERBAL_TOTAL_BV", "ADOS_GOTHAM_SOCAFFECT", "ADOS_2_SOCAFFECT", "ADOS_GOTHAM_RRB", "ADOS_2_RRB", "SRS_MOTIVATION", "SRS_MOTIVATION_RAW", "VINELAND_DAILYLIVING_STANDARD", "VINELAND_DAILYLVNG_STANDARD", "VINELAND_COPING_V_SCALED"]
    phenotypes = ["SRS_MOTIVATION", "SRS_MOTIVATION_RAW", "SRS_COMMUNICATION", "SRS_COMMUNICATION_RAW", "VINELAND_DAILYLIVING_STANDARD", "VINELAND_DAILYLVNG_STANDARD", "BRIEF_GEC_T"]
    participants_df = pd.read_csv(op.join(group_path, "sub-group_task-rest_desc-1S2StTesthabenula_table.txt"), sep="\t")
    unique_subject_count = participants_df['Subj'].nunique()
    print(f"Number of unique subjects: {unique_subject_count}", flush=True)
 

    # List to store subject details
    subject_pheno = []

    # Iterate through the phenotype columns
    for phenotype in phenotypes:
        # Check if the column exists in the dataframe
        if (phenotype in phenotypic_df.columns):
            # Iterate through the rows to check if the subject has a value in the phenotype column
            for idx, row in phenotypic_df.iterrows():
                if pd.notna(row[phenotype]):  # Check if the value is not NaN
                    subject_id = row["participant_id"]  # Assuming the subject ID column is named "participant_id"
                    score = row[phenotype]  # Get the score for the phenotype
                    # Append a dictionary with subject details
                    subject_pheno.append({
                        "subject_id": subject_id,
                        "phenotype": phenotype,
                        "score": score
                    })

    # Convert to DataFrame to handle duplicates based on 'subject_id' and 'phenotype'
    subject_pheno_df = pd.DataFrame(subject_pheno).drop_duplicates(subset=['subject_id', 'phenotype'])

    results = []

    # Iterate through the subjects in participants_df
    for subject in participants_df['Subj']:
        if subject in subject_pheno_df['subject_id'].values:  # Check if the subject is in the subject_ids list
            subject_path = os.path.join(rsfc_dir, subject, "func")

            # Check if the subject path exists
            if not os.path.exists(subject_path):
                print(f"Skipping subject {subject}: Path does not exist.")
                continue

            subject_brik = op.join(
                subject_path,
                f"{subject}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-norm_bucketREML+tlrc.BRIK[1]",
            )
            subject_task_brik = op.join(
                subject_path,
                f"{subject}_task-rest_run-1_space-MNI152NLin2009cAsym_res-2_desc-norm_bucketREML+tlrc.BRIK[1]",
            )
            subject_nii = os.path.join(
                subject_path,
                f"{subject}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-norm_zmap.nii.gz",
            )

            if not os.path.exists(subject_nii): #this part of the code is a little finicky, have to change it
                afni2nifti(subject_brik, subject_nii)

            print(f"\t\t\tProcessing cluster {cluster_id} ROI for {subject}", flush=True)
            
            # Load cluster ROI mask
            cluster_mask = nib.load(roi_file)
            
            # Create masker with cluster ROI
            masker = NiftiMasker(mask_img=cluster_mask)
            zscores = masker.fit_transform(subject_nii)

            # Average across all voxels in the cluster ROI
            zscore = np.mean(zscores)
            print(f"\t\t\tCluster {cluster_id} mean z-score: {zscore:.4f}", flush=True)

            age = participants_df.loc[participants_df['Subj'] == subject, 'age'].values[0]
            group = participants_df.loc[participants_df['Subj'] == subject, 'group'].values[0]

            results.append(
                {"Subject": subject, "Group": group, "Age": age, "Cluster": cluster_id, "Correlation": zscore}
            )
                    
    corr_df = pd.DataFrame(results)


    # Merge the subject_pheno_df and corr_df
    merged_df = pd.merge(corr_df, subject_pheno_df, left_on='Subject', right_on='subject_id')
    merged_df = merged_df.drop(columns=['subject_id'])  # Remove the 'subject_id' column

    output_file = os.path.join(group_dir, f"pheno-correlation-cluster{cluster_id}.csv")
    merged_df.to_csv(output_file, index=False)
    print(f"\nSaved results to {output_file}", flush=True)



def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()