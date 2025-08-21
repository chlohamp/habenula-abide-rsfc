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

    return clean_briks_files, clean_mask_files


def write_table(table_fn_file):
    tab_labels = [
        "Subj",
        "group",
        "site",
        "age",
        "gender",
        "medication",
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
    medication = sub_df["CURRENT_MED_STATUS"].values[0]
    InputFile = "{brik}[{idx}]".format(brik=subjAve_roi_briks_file, idx=idx)

    group = int(float(group))
    group = "asd" if group == 1 else "td" if group == 2 else group

    medication = int(float(medication))

    cov_variables = [
        subject,
        group,
        site,
        age,
        gender,
        medication,
        InputFile,
    ]

    cov_variables_str = [str(x) for x in cov_variables]
    with open(table_fn_file, "a") as fo:
        fo.write("{}\n".format("\t".join(cov_variables_str)))


def run_lmer(bucket_fn, mask_fn, table_file, n_jobs):
    #model = "'group+age+gender+(1|site)'"
    model = "'group+age+gender+medication+(1|site)'"

    asd_mean = "asd_mean 'group : 1*asd'"
    td_mean = "td_mean 'group : 1*td'"
    group_mean = "group_mean 'group : 0.5*asd +0.5*td'"
    group_diff = "td-asd 'group : 1*asd -1*td'"
    cmd = f"3dLMEr -prefix {bucket_fn} \
        -mask {mask_fn} \
        -model {model} \
        -qVars 'age' \
        -qVarCenters '0' \
        -gltCode {asd_mean} \
        -gltCode {td_mean} \
        -gltCode {group_mean} \
        -gltCode {group_diff} \
        -resid {bucket_fn}_res \
        -dbgArgs \
        -jobs {n_jobs} \
        -dataTable @{table_file}"

    print(f"\t\t{cmd}", flush=True)
    os.system(cmd)


'''def roi_clst(bucket_fn, template, ClusterMap, ClusterEffEst):
    cmd = f"3dClusterize \
        -inset {bucket_fn} \
        -mask {template} \
        -ithr 11 \
        -idat 10 \
        -bisided p=0.001 \
        -NN 2 \
        -clust_nvox 100            \
        -pref_map {ClusterMap} \
        -pref_dat {ClusterEffEst}"
    
    print(f"\t\t{cmd}", flush=True)
    os.system(cmd)

def calc_roi(ClusterMap, cluster_id, output_file):
    cmd = f"3dcalc \
        -a {ClusterMap} \
        -expr 'equals(a,{cluster_id})' \
        -prefix {output_file} \
        -datum byte"
    print(f"\t\t{cmd}", flush=True)
    os.system(cmd)
'''


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

    participants_df = pd.read_csv(op.join(dset, "participants.tsv"), sep="\t")
    #participants_df = participants_df[(participants_df["AGE_AT_SCAN"] >= 5) & (participants_df["AGE_AT_SCAN"] <= 21)]

    # Define directories
    rsfc_subjs_dir = op.join(rsfc_dir, "**", "func")
    rsfc_group_dir = op.join(rsfc_dir, "group-medication_rerun")
    os.makedirs(rsfc_group_dir, exist_ok=True)

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

    # Remove outliers using MRIQC metrics
    clean_briks_files, clean_mask_files = remove_ouliers(
        mriqc_dir, briks_files, mask_files
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
                "CURRENT_MED_STATUS",
            ]
        ],
        clean_briks_files,
        clean_mask_files,
    )
    print(
        f"Removing missing data: {len(clean_briks_files)}/{len(briks_files)}",
        flush=True,
    )
    assert len(clean_briks_files) == len(clean_mask_files)

    # Write group file
    clean_briks_fn = op.join(
        rsfc_group_dir,
        f"sub-group_task-rest_space-{space}_briks.txt",
    )
    if not op.exists(clean_briks_fn):
        with open(clean_briks_fn, "w") as fo:
            for tmp_brik_fn in clean_briks_files:
                fo.write(f"{tmp_brik_fn}\n")

    # Create group mask
    group_mask_fn = op.join(
        rsfc_group_dir,
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

    roi_dir = op.join(rsfc_group_dir, roi)
    os.makedirs(roi_dir, exist_ok=True)

    # Conform table_fn
    write_new_table = False
    table_fn = op.join(roi_dir, f"sub-group_task-rest_desc-1S2StTest{roi}_table.txt")
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
    # Whole-brain, one-sample t-tests, and two-sample t-tests
    onetwottest_briks_fn = op.join(
        roi_dir, f"sub-group_task-rest_desc-1S2StTest{roi}_briks"
    )

    os.chdir(op.dirname(onetwottest_briks_fn))
    if not op.exists(f"{onetwottest_briks_fn}+tlrc.BRIK"):
        run_lmer(
            op.basename(onetwottest_briks_fn),
            group_mask_fn,
            table_fn,
            n_jobs,
        )

    '''onetwottest_briks_map = op.join(
        roi_dir, f"sub-group_task-rest_desc-1S2StTest{roi}"
    )
    onetwottest_briks_eff = op.join(
            roi_dir, f"sub-group_task-rest_desc-1S2StTest{roi}"
        )

    if not op.exists(f"{onetwottest_briks_map}+tlrc.BRIK"):
        roi_clst(
            f"{onetwottest_briks_fn}+tlrc.BRIK",
            group_mask_fn,
            onetwottest_briks_map,
            onetwottest_briks_eff
        )

    for cluster in range(1, 4):  # Loop over the number of clusters
        roi_file = f"sub-group_task-rest_desc-diff{roi}roi{cluster}.nii.gz"
        calc_roi(
            f"{onetwottest_briks_map}+tlrc.BRIK",
            cluster,
            roi_file
        )'''


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
