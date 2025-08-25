#!/bin/bash

# Configuration - Server paths
T1W_BASE_PATH="/home/data/nbc/Laird_ABIDE/dset/derivatives/fmriprep-23.1.3"

# Input and output paths
PARTICIPANTS_FILE="group-participants.tsv"  # Copy this file to server first
OUTPUT_DIR="/home/champ007/t1w_files_for_transfer"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Counters
found=0
missing=0
copied=0

echo "Copying T1w files for group participants..."
echo "Looking in: $T1W_BASE_PATH"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Read subjects (skip header)
tail -n +2 "$PARTICIPANTS_FILE" | cut -f1 | while read subject; do
    # T1w filename pattern
    t1w_file="${subject}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz"
    
    # T1w file should be in: base_path/subject/anat/filename
    t1w_path="$T1W_BASE_PATH/$subject/anat/$t1w_file"
    
    if [ -f "$t1w_path" ]; then
        echo "Found: $t1w_path"
        cp "$t1w_path" "$OUTPUT_DIR/"
        if [ $? -eq 0 ]; then
            ((copied++))
            ((found++))
        else
            echo "Error copying $t1w_path"
            ((missing++))
        fi
    else
        echo "Missing: $t1w_file"
        ((missing++))
    fi
done

echo ""
echo "Summary:"
echo "Files found and copied: $copied"
echo "Files missing: $missing"
echo ""

if [ $copied -gt 0 ]; then
    echo "Files copied to: $OUTPUT_DIR"
    echo ""
    echo "To create a zip file for transfer:"
    echo "cd $(dirname $OUTPUT_DIR)"
    echo "zip -r t1w_files.zip $(basename $OUTPUT_DIR)/"
fi
