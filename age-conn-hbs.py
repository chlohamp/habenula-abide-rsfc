import argparse
import os
import os.path as op
from glob import glob

import nibabel as nib
import numpy as np
import pandas as pd


def _get_parser():
    parser = argparse.ArgumentParser(description="Extract age and habenula connectivity for age-effects ROIs")
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
        "--age_dir",
        dest="age_dir",
        required=True,
        help="Path to directory containing age-effects cluster ROI masks",
    )
    parser.add_argument(
        "--cluster_id",
        dest="cluster_id",
        required=True,
        help="Cluster ID to process",
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
    age_dir,
    cluster_id,
    n_jobs,
):
    """Extract age and habenula connectivity for age-effects ROIs."""
    os.system(f"export OMP_NUM_THREADS={n_jobs}")
    roi_dict = {label: x * 3 + 1 for x, label in enumerate(roi_lst)}
    print(roi_dict, flush=True)
    space = "MNI152NLin2009cAsym"
    n_jobs = int(n_jobs)

    # Load the specified cluster ROI mask from age-effects directory (directly in age_dir)
    roi_file = op.join(age_dir, f"cluster_{cluster_id.zfill(2)}_roi.nii.gz")
    if not op.exists(roi_file):
        raise FileNotFoundError(f"Cluster {cluster_id} ROI not found at {roi_file}")
    
    print(f"Processing age-effects cluster {cluster_id} ROI: {roi_file}", flush=True)

    group_path = os.path.join(group_dir, roi)
    
    # Load participant data from the age-effects analysis table (in habenula subdirectory)
    age_table_path = op.join(age_dir, roi, "sub-group_task-rest_desc-AgeEffhabenula_table.txt")
    if not op.exists(age_table_path):
        raise FileNotFoundError(f"Age-effects table not found at {age_table_path}")
    
    participants_df = pd.read_csv(age_table_path, sep="\t")
    unique_subject_count = participants_df['Subj'].nunique()
    print(f"Number of unique subjects in age-effects analysis: {unique_subject_count}", flush=True)

    results = []

    # Iterate through the subjects in participants_df
    for idx, row in participants_df.iterrows():
        subject = row['Subj']
        age = row['age']
        group = row['group']
        
        subject_path = os.path.join(rsfc_dir, subject, "func")

        # Check if the subject path exists
        if not os.path.exists(subject_path):
            print(f"Skipping subject {subject}: Path does not exist.", flush=True)
            continue

        # Try multiple naming conventions for BRIK files (no run, run-1, run-2, run-3)
        brik_patterns = [
            (f"{subject}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-norm_bucketREML+tlrc.BRIK", "task-rest"),
            (f"{subject}_task-rest_run-1_space-MNI152NLin2009cAsym_res-2_desc-norm_bucketREML+tlrc.BRIK", "task-rest_run-1"),
            (f"{subject}_task-rest_run-2_space-MNI152NLin2009cAsym_res-2_desc-norm_bucketREML+tlrc.BRIK", "task-rest_run-2"),
            (f"{subject}_task-rest_run-3_space-MNI152NLin2009cAsym_res-2_desc-norm_bucketREML+tlrc.BRIK", "task-rest_run-3"),
        ]
        
        # Find which BRIK file exists
        brik_to_use = None
        nii_suffix = None
        for brik_name, suffix in brik_patterns:
            brik_path = op.join(subject_path, brik_name)
            if op.exists(brik_path):
                brik_to_use = f"{brik_path}[1]"
                nii_suffix = suffix
                break
        
        if brik_to_use is None:
            print(f"Skipping subject {subject}: BRIK file not found.", flush=True)
            continue
        
        subject_nii = os.path.join(
            subject_path,
            f"{subject}_{nii_suffix}_space-MNI152NLin2009cAsym_res-2_desc-norm_zmap.nii.gz",
        )

        # Check if NIfTI exists and is valid, otherwise create it
        needs_conversion = False
        if not os.path.exists(subject_nii):
            needs_conversion = True
        else:
            # Check if existing file is corrupted
            try:
                test_img = nib.load(subject_nii)
                _ = test_img.get_fdata()  # Try to load data to check for corruption
            except Exception as e:
                print(f"\t\t\tNIfTI file is corrupted for {subject}, will recreate", flush=True)
                os.remove(subject_nii)
                needs_conversion = True
        
        if needs_conversion:
            print(f"\t\t\tConverting BRIK to NIfTI for {subject}", flush=True)
            afni2nifti(brik_to_use, subject_nii)
            
            # Verify the NIfTI file was created successfully and is valid
            if not os.path.exists(subject_nii):
                print(f"Skipping subject {subject}: Failed to create NIfTI file.", flush=True)
                continue
            
            # Check if the newly created file is valid
            try:
                test_img = nib.load(subject_nii)
                _ = test_img.get_fdata()
            except Exception as e:
                print(f"Skipping subject {subject}: Newly created NIfTI is invalid ({str(e)})", flush=True)
                if os.path.exists(subject_nii):
                    os.remove(subject_nii)
                continue

        print(f"\t\t\tProcessing age-effects cluster {cluster_id} ROI for {subject}", flush=True)
        
        try:
            # Load cluster ROI mask and subject connectivity map
            cluster_mask_img = nib.load(roi_file)
            cluster_mask_data = cluster_mask_img.get_fdata()
            
            subject_img = nib.load(subject_nii)
            subject_data = subject_img.get_fdata()
            
            # Binary mask - extract all voxels > 0
            binary_mask = cluster_mask_data > 0
            n_voxels = np.sum(binary_mask)
            
            if n_voxels == 0:
                print(f"\t\t\tWARNING: No voxels in cluster {cluster_id}!", flush=True)
                continue
            
            # Extract values from subject data where mask > 0
            masked_values = subject_data[binary_mask]
            mean_zscore = np.mean(masked_values)
            print(f"\t\t\tCluster {cluster_id} mean z-score: {mean_zscore:.4f} (n_voxels={n_voxels})", flush=True)

            # Extract sex and site from participants_df (matching pheno-conn-hbs.py pattern)
            sex = participants_df.loc[participants_df['Subj'] == subject, 'gender'].values[0]
            site = participants_df.loc[participants_df['Subj'] == subject, 'site'].values[0]

            results.append({
                "Subject": subject,
                "Group": group,
                "Age": age,
                "Sex": sex,
                "Site": site,
                "Cluster": cluster_id,
                "Mean_Zscore": mean_zscore,
                "N_Voxels": n_voxels
            })
        except Exception as e:
            print(f"Error processing subject {subject}: {str(e)}", flush=True)
            continue
                
    # Create DataFrame with results
    results_df = pd.DataFrame(results)

    # Save results
    output_file = os.path.join(age_dir, f"age-connectivity-cluster{cluster_id}.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved results to {output_file}", flush=True)
    print(f"Total subjects processed: {len(results_df)}", flush=True)


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
