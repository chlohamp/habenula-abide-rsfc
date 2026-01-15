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


def _get_parser():
    parser = argparse.ArgumentParser(description="Run group analysis")
    parser.add_argument(
        "--dset",
        dest="dset",
        required=True,
        help="Path to BIDS directory",
    )
    parser.add_argument(
        "--mriqc_dir",
        dest="mriqc_dir",
        required=True,
        help="Path to MRIQC directory",
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
        "--n_jobs",
        dest="n_jobs",
        required=True,
        help="CPUs",
    )
    return parser


def afni2nifti(afni_fn, nifti_fn):
    cmd = f"3dAFNItoNIFTI \
                -prefix {nifti_fn} \
                {afni_fn}"
    print(f"\t\t\t{cmd}", flush=True)
    os.system(cmd)


def conn_resample(roi_in, roi_out, template):

    cmd = f"3dresample \
            -prefix {roi_out} \
            -master {template} \
            -inset {roi_in}"
    print(f"\t\t\t{cmd}", flush=True)
    os.system(cmd)


def remove_ouliers(mriqc_dir, briks_files, mask_files):

    runs_to_exclude_df = pd.read_csv(
        op.join(mriqc_dir, "runs_to_exclude.tsv"), sep="\t"
    )
    runs_to_exclude = runs_to_exclude_df["bids_name"].tolist()
    prefixes_tpl = tuple(runs_to_exclude)

    clean_briks_files = [
        x for x in briks_files if not op.basename(x).startswith(prefixes_tpl)
    ]
    clean_mask_files = [
        x for x in mask_files if not op.basename(x).startswith(prefixes_tpl)
    ]

    return clean_briks_files, clean_mask_files


def remove_missingdat(participants_df, briks_files, mask_files):
    participants_df = participants_df.replace(["-9999", "`", -9999, 999, 777], np.nan)
    participants_df = participants_df.dropna()
    subjects_to_keep = participants_df["participant_id"].tolist()

    prefixes_tpl = tuple(subjects_to_keep)

    clean_briks_files = [
        x for x in briks_files if op.basename(x).startswith(prefixes_tpl)
    ]
    clean_mask_files = [
        x for x in mask_files if op.basename(x).startswith(prefixes_tpl)
    ]
    
    # Ensure 1-to-1 matching between briks and masks
    # Extract base identifiers (subject_session_run) from briks files
    briks_bases = set()
    for brik in clean_briks_files:
        # Remove the suffix after 'desc-norm_bucketREML+tlrc.BRIK'
        base = op.basename(brik).replace('_desc-norm_bucketREML+tlrc.BRIK', '')
        briks_bases.add(base)
    
    # Filter masks to only those with matching briks
    matched_mask_files = []
    for mask in clean_mask_files:
        # Remove the suffix after 'desc-brain_mask.nii.gz'
        base = op.basename(mask).replace('_desc-brain_mask.nii.gz', '')
        if base in briks_bases:
            matched_mask_files.append(mask)
    
    clean_mask_files = matched_mask_files

    return clean_briks_files, clean_mask_files


def write_table(table_fn_file):
    tab_labels = [
        "Subj",
        "group",
        "site",
        "age",
        "gender",
        "InputFile",
    ]
    with open(table_fn_file, "w") as fo:
        fo.write("{}\n".format("\t".join(tab_labels)))


def append2table(subject, subjAve_roi_briks_file, idx, participants_df, table_fn_file):
    sub_df = participants_df[participants_df["participant_id"] == subject]

    sub_df = sub_df.fillna("")
    group = sub_df["DX_GROUP"].values[0]
    site = sub_df["SITE_ID"].values[0]
    age = sub_df["AGE_AT_SCAN"].values[0]
    gender = sub_df["SEX"].values[0]
    InputFile = "{brik}[{idx}]".format(brik=subjAve_roi_briks_file, idx=idx)

    group = int(float(group))
    group = "asd" if group == 1 else "td" if group == 2 else group

    cov_variables = [
        subject,
        group,
        site,
        age,
        gender,
        InputFile,
    ]

    cov_variables_str = [str(x) for x in cov_variables]
    with open(table_fn_file, "a") as fo:
        fo.write("{}\n".format("\t".join(cov_variables_str)))


def run_agelmer(bucket_fn, mask_fn, center, table_file, n_jobs):
    data_table = pd.read_csv(table_file, sep='\t')
    model = "'group*age+gender+(1|site)'"

    asd_mean = "asd_mean 'group : 1*asd age :'"
    td_mean = "td_mean 'group : 1*td age :'"
    group = "group 'group : 1*asd &1*td age :'"
    group_mean = "group_mean 'group : 0.5*asd +0.5*td age :'"
    group_diff = "asd-td  'group : 1*asd -1*td age :'"
    group_by_age_interaction = "group_by_age 'group : 1*asd -1*td age : 1'"

    

    cmd = f"3dLMEr -prefix {bucket_fn} \
        -mask {mask_fn} \
        -model {model} \
        -qVars 'age' \
        -qVarCenters {center} \
        -gltCode {asd_mean} \
        -gltCode {td_mean} \
        -gltCode {group} \
        -gltCode {group_mean} \
        -gltCode {group_diff} \
        -gltCode {group_by_age_interaction} \
        -resid {bucket_fn}_res \
        -dbgArgs \
        -jobs {n_jobs} \
        -dataTable @{table_file}"

    print(f"\t\t{cmd}", flush=True)
    os.system(cmd)

def main(
    dset,
    mriqc_dir,
    rsfc_dir,
    template,
    template_mask,
    roi_lst,
    roi,
    n_jobs,
):
    """Run group analysis workflows on a given dataset."""
    os.system(f"export OMP_NUM_THREADS={n_jobs}")
    roi_dict = {label: x * 3 + 1 for x, label in enumerate(roi_lst)}
    print(roi_dict, flush=True)
    space = "MNI152NLin2009cAsym"
    n_jobs = int(n_jobs)

    participants_df = pd.read_csv(op.join(dset, "participants.tsv"), sep="\t", low_memory=False)
    # Filter participants by age
    participants_df = participants_df[(participants_df["AGE_AT_SCAN"] >= 5) & (participants_df["AGE_AT_SCAN"] <= 21)]
    average_age = participants_df["AGE_AT_SCAN"].mean()
    print(average_age)

    # Define directories
    rsfc_subjs_dir = op.join(rsfc_dir, "**", "func")
    rsfc_age_dir = op.join(rsfc_dir, "age-effect5-21") #changed to look at age
    os.makedirs(rsfc_age_dir, exist_ok=True)

    # Collect important files
    briks_files = sorted(
        glob(
            op.join(
                rsfc_subjs_dir,
                f"*task-rest*_space-{space}*_desc-norm_bucketREML+tlrc.BRIK",
            ),
            recursive=True,
        )
    )
    mask_files = sorted(
        glob(
            op.join(
                rsfc_subjs_dir, f"*task-rest*_space-{space}*_desc-brain_mask.nii.gz"
            ),
            recursive=True,
        )
    )
    
    # Select only first run per subject (some have multiple runs/sessions)
    subject_briks = {}
    for brik in briks_files:
        subject = op.basename(brik).split("_")[0]
        if subject not in subject_briks:
            subject_briks[subject] = brik
    briks_files = sorted(list(subject_briks.values()))
    
    subject_masks = {}
    for mask in mask_files:
        subject = op.basename(mask).split("_")[0]
        if subject not in subject_masks:
            subject_masks[subject] = mask
    mask_files = sorted(list(subject_masks.values()))
    
    print(
        f"After selecting first run per subject: {len(briks_files)} briks, {len(mask_files)} masks",
        flush=True,
    )

    # Remove outliers using MRIQC metrics
    clean_briks_files, clean_mask_files = remove_ouliers(
        mriqc_dir, briks_files, mask_files
    )
    print(
        f"After removing outliers: {len(clean_briks_files)} briks, {len(clean_mask_files)} masks",
        flush=True,
    )

    # Remove missing data in covariates:
    clean_briks_files, clean_mask_files = remove_missingdat(
        participants_df[
            [
                "participant_id",
                "DX_GROUP",
                "SITE_ID",
                "AGE_AT_SCAN",
                "SEX",
            ]
        ],
        clean_briks_files,
        clean_mask_files,
    )
    print(
        f"After removing missing data: {len(clean_briks_files)} briks, {len(clean_mask_files)} masks (started with {len(briks_files)})",
        flush=True,
    )
    assert len(clean_briks_files) == len(clean_mask_files), f"Mismatch: {len(clean_briks_files)} briks != {len(clean_mask_files)} masks"

    # Write group file
    clean_briks_fn = op.join(
        rsfc_age_dir,
        f"sub-group_task-rest_space-{space}_briks.txt",
    )
    if not op.exists(clean_briks_fn):
        with open(clean_briks_fn, "w") as fo:
            for tmp_brik_fn in clean_briks_files:
                fo.write(f"{tmp_brik_fn}\n")

    # Create group mask
    group_mask_fn = op.join(
        rsfc_age_dir,
        f"sub-group_task-rest_space-{space}_desc-brain_mask.nii.gz",
    )
    if not op.exists(group_mask_fn):
        if template_mask is None:
            template_mask_img = nib.load(clean_mask_files[0])
        else:
            template_mask_img = nib.load(template_mask)
        for clean_mask_file in clean_mask_files:
            clean_mask_img = nib.load(clean_mask_file)
            if clean_mask_img.shape != template_mask_img.shape:
                clean_res_mask_img = image.resample_to_img(
                    clean_mask_img, template_mask_img, interpolation="nearest"
                )
                nib.save(clean_res_mask_img, clean_mask_file)

        group_mask = masking.intersect_masks(clean_mask_files, threshold=0.5)
        nib.save(group_mask, group_mask_fn)

    # Get template
    if template is None:
        # Resampling group to one subject
        clean_briks_file = clean_briks_files[0]
        template = op.join(f"{clean_briks_file}'[{roi_dict[roi]}]'")
        template_img = nib.load(clean_briks_file)
    else:
        template_img = nib.load(template)
    print(f"Using template {template} with size: {template_img.shape}", flush=True)

    roi_dir = op.join(rsfc_age_dir, roi)
    os.makedirs(roi_dir, exist_ok=True)

    # Conform table_fn
    write_new_table = False
    table_fn = op.join(roi_dir, f"sub-group_task-rest_desc-AgeEff{roi}_table.txt")
    if not op.exists(table_fn):
        write_table(table_fn)
        write_new_table = True

    # Calculate subject and ROI level average connectivity
    subjects = [op.basename(x).split("_")[0] for x in clean_briks_files]
    subjects = list(set(subjects))
    print(f"Group analysis sample size: {len(subjects)}")

    for subject in subjects:
        subj_briks_files = [x for x in clean_briks_files if subject in x]
        # assert len(subj_briks_files) == 1

        # For this project there is only one run and session per subject
        # Actually some subject contain multiple session. Select the first one
        subj_briks_file = subj_briks_files[0]

        rsfc_subj_dir = op.join(rsfc_dir, subject, "func")
        prefix = op.basename(subj_briks_file).split("space-")[0].rstrip("_")

        subj_briks_res_file = op.join(
            rsfc_subj_dir,
            f"{prefix}_space-{space}_desc-{roi}res_coef",
        )

        # Resampling to template fov is different
        subj_roi_briks = nib.load(subj_briks_file)
        if not _check_same_fov(subj_roi_briks, reference_masker=template_img):
            if not op.exists(f"{subj_briks_res_file}+tlrc.BRIK"):
                conn_resample(
                    subj_briks_file,
                    subj_briks_res_file,
                    template,
                )
            subj_briks_file = f"{subj_briks_res_file}+tlrc.BRIK"

        # Append subject specific info for table_fn
        if op.exists(table_fn) and write_new_table:
            append2table(
                subject, subj_briks_file, roi_dict[roi], participants_df, table_fn
            )

    # Statistical analysis
    # age interactions
    age_briks_fn = op.join(
        roi_dir, f"sub-group_task-rest_desc-AgeEff{roi}_briks"
    )

    os.chdir(op.dirname(age_briks_fn))
    if not op.exists(f"{age_briks_fn}+tlrc.BRIK"):
        run_agelmer(
            op.basename(age_briks_fn),
            group_mask_fn,
            average_age,
            table_fn,
            n_jobs,
        )


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
