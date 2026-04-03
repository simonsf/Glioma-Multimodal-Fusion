#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8
"""
Custom code for functional connectivity (FC) model feature engineering in glioma multimodal imaging analysis
Key procedures:
1. FA value statistics for tumor and edema regions
2. Edema region segmentation into high/low FA sub-regions
3. Skull-stripping of T1 images in DWI space
4. Spatial co-registration of DTI/fMRI to T1WI template
5. Generation of 4-region atlas (tumor, edema_lowFA, edema_highFA, normal brain)
6. Calculation of 4×4 functional connectivity matrix
"""

import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd
import ants
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure

# -------------------------- Configuration Parameters --------------------------
# Original input directory (keep the exact path from original code)
INPUT_ROOT_DIR = ".../input/brain_mask/"
# Independent output directory (avoid polluting raw data)
OUTPUT_ROOT_DIR = ".../results/FC_output/"
# Specify 3 test patients (replace with actual patient folder names)
TEST_PATIENTS = ['Patient01', 'Patient02', 'Patient03']

# Ensure output directory exists
os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)


# -------------------------- 1. FA Statistics Calculation --------------------------
def calculate_fa_statistics():
    """
    Calculate maximum, minimum, and median FA values for tumor and edema regions
    across specified test patients
    """
    print("=== Step 1: Calculating FA Statistics for Tumor/Edema Regions ===")
    regions = {
        "tumor": "T1_tumor_pair_dwi_space.nii.gz",
        "edema": "T1_edema_dwi_space.nii.gz"
    }
    all_results = []

    for patient_name in TEST_PATIENTS:
        input_folder = os.path.join(INPUT_ROOT_DIR, patient_name)
        output_folder = os.path.join(OUTPUT_ROOT_DIR, patient_name)
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\nProcessing patient: {patient_name}")
        
        # Check patient folder existence
        if not os.path.exists(input_folder):
            print(f"Warning: Patient folder {input_folder} does not exist, skipped")
            continue

        # Define FA image path
        fa_path = os.path.join(input_folder, "fa.nii.gz")
        if not os.path.exists(fa_path):
            print(f"Warning: FA file missing for {patient_name}, skipped")
            continue

        try:
            # Load FA data
            fa_img = nib.load(fa_path)
            fa_data = fa_img.get_fdata()
            patient_results = {"patient_name": patient_name}

            # Process each region
            for region_name, mask_filename in regions.items():
                mask_path = os.path.join(input_folder, mask_filename)
                if not os.path.exists(mask_path):
                    print(f"Warning: {region_name} mask missing for {patient_name}, skipped")
                    continue

                # Load mask data
                mask_img = nib.load(mask_path)
                mask_data = mask_img.get_fdata()

                # Check dimension matching
                if fa_data.shape != mask_data.shape:
                    print(f"Warning: FA and {region_name} mask dimensions mismatch, skipped")
                    continue

                # Extract FA values in region
                region_mask = mask_data > 0
                region_fa = fa_data[region_mask]

                if len(region_fa) == 0:
                    print(f"Warning: No valid voxels in {region_name} mask for {patient_name}, skipped")
                    continue

                # Calculate statistics
                max_fa = np.max(region_fa)
                min_fa = np.min(region_fa)
                median_fa = np.median(region_fa)

                patient_results[f"{region_name}_max_fa"] = max_fa
                patient_results[f"{region_name}_min_fa"] = min_fa
                patient_results[f"{region_name}_median_fa"] = median_fa

                print(f"{region_name} FA values - Max: {max_fa:.4f}, Min: {min_fa:.4f}, Median: {median_fa:.4f}")

            all_results.append(patient_results)

        except Exception as e:
            print(f"Error processing {patient_name}: {str(e)}, skipped")
            continue

    # Save statistics to CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_csv = os.path.join(OUTPUT_ROOT_DIR, "fa_statistics_summary.csv")
        results_df.to_csv(output_csv, index=False)
        print(f"\nFA statistics saved to: {output_csv}")


# -------------------------- 2. Edema Segmentation by FA --------------------------
def segment_edema_by_fa():
    """
    Segment edema region into high-FA and low-FA sub-regions using individual median FA threshold
    """
    print("\n=== Step 2: Segmenting Edema into High/Low FA Regions ===")
    processed = 0
    skipped = 0

    for patient_name in TEST_PATIENTS:
        input_folder = os.path.join(INPUT_ROOT_DIR, patient_name)
        output_folder = os.path.join(OUTPUT_ROOT_DIR, patient_name)
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\nProcessing patient: {patient_name}")
        
        if not os.path.exists(input_folder):
            print(f"Warning: Patient folder {input_folder} does not exist, skipped")
            skipped += 1
            continue

        # Define file paths
        fa_path = os.path.join(input_folder, "fa.nii.gz")
        edema_mask_path = os.path.join(input_folder, "T1_edema_dwi_space.nii.gz")

        # Check file existence
        missing_files = []
        if not os.path.exists(fa_path):
            missing_files.append("fa.nii.gz")
        if not os.path.exists(edema_mask_path):
            missing_files.append("T1_edema_dwi_space.nii.gz")

        if missing_files:
            print(f"Warning: Missing files {missing_files}, skipped")
            skipped += 1
            continue

        try:
            # Load image data
            fa_img = nib.load(fa_path)
            edema_mask_img = nib.load(edema_mask_path)

            fa_data = fa_img.get_fdata()
            edema_mask_data = edema_mask_img.get_fdata()

            # Check dimension matching
            if fa_data.shape != edema_mask_data.shape:
                print(f"Error: FA and edema mask dimensions mismatch, skipped")
                skipped += 1
                continue

            # Extract edema FA values
            edema_mask = edema_mask_data > 0
            edema_fa_values = fa_data[edema_mask]

            if len(edema_fa_values) == 0:
                print(f"Warning: No valid voxels in edema mask, skipped")
                skipped += 1
                continue

            # Calculate median threshold
            fa_threshold = np.median(edema_fa_values)
            print(f"Edema FA median threshold: {fa_threshold:.4f}")

            # Generate high/low FA masks
            highfa_mask_data = np.where((edema_mask) & (fa_data >= fa_threshold), 1, 0).astype(np.int16)
            lowfa_mask_data = np.where((edema_mask) & (fa_data < fa_threshold), 1, 0).astype(np.int16)

            # Create NIfTI images
            highfa_img = nib.Nifti1Image(highfa_mask_data, edema_mask_img.affine, edema_mask_img.header)
            lowfa_img = nib.Nifti1Image(lowfa_mask_data, edema_mask_img.affine, edema_mask_img.header)

            # Save masks to output directory
            highfa_save_path = os.path.join(output_folder, "edema_highFA_mask.nii.gz")
            lowfa_save_path = os.path.join(output_folder, "edema_lowFA_mask.nii.gz")

            nib.save(highfa_img, highfa_save_path)
            nib.save(lowfa_img, lowfa_save_path)

            print(f"Saved high-FA mask: {highfa_save_path}")
            print(f"Saved low-FA mask: {lowfa_save_path}")
            processed += 1

        except Exception as e:
            print(f"Error processing {patient_name}: {str(e)}, skipped")
            skipped += 1

    print(f"\nSegmentation completed - Processed: {processed}, Skipped: {skipped}")


# -------------------------- 3. T1 Skull-Stripping (DWI Space) --------------------------
def skull_strip_t1_dwi():
    """
    Perform skull-stripping on T1 images in DWI space using brain mask
    """
    print("\n=== Step 3: Skull-Stripping T1 Images in DWI Space ===")

    for patient_name in TEST_PATIENTS:
        input_folder = os.path.join(INPUT_ROOT_DIR, patient_name)
        output_folder = os.path.join(OUTPUT_ROOT_DIR, patient_name)
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\nProcessing patient: {patient_name}")
        
        if not os.path.exists(input_folder):
            print(f"Warning: Patient folder {input_folder} does not exist, skipped")
            continue

        # Define file paths
        t1_path = os.path.join(input_folder, "T1_dwi.nii.gz")
        mask_path = os.path.join(input_folder, "brain_mask_dwi.nii.gz")

        if not os.path.exists(t1_path):
            print(f"Warning: T1_dwi.nii.gz missing for {patient_name}, skipped")
            continue
        if not os.path.exists(mask_path):
            print(f"Warning: brain_mask_dwi.nii.gz missing for {patient_name}, skipped")
            continue

        try:
            # Load T1 and mask data
            t1_img = nib.load(t1_path)
            mask_img = nib.load(mask_path)

            t1_data = t1_img.get_fdata()
            mask_data = mask_img.get_fdata()

            # Apply skull-stripping mask
            mask_binary = np.where(mask_data > 0, 1, 0)
            t1_skull_data = t1_data * mask_binary

            # Save skull-stripped T1
            t1_skull_img = nib.Nifti1Image(t1_skull_data, t1_img.affine, t1_img.header)
            output_path = os.path.join(output_folder, "T1_dwi_skull.nii.gz")
            nib.save(t1_skull_img, output_path)

            print(f"Generated skull-stripped T1: {output_path}")

        except Exception as e:
            print(f"Error processing {patient_name}: {str(e)}, skipped")


# -------------------------- 4. Spatial Co-Registration (DWI → fMRI) --------------------------
def register_dwi_to_fmri():
    """
    Perform spatial co-registration of DWI-space T1 to fMRI space (T1WI template)
    and transform edema masks to fMRI space
    """
    print("\n=== Step 4: Spatial Co-Registration (DWI → fMRI Space) ===")

    for patient_name in TEST_PATIENTS:
        input_folder = os.path.join(INPUT_ROOT_DIR, patient_name)
        output_folder = os.path.join(OUTPUT_ROOT_DIR, patient_name)
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\nProcessing patient: {patient_name}")
        
        if not os.path.exists(input_folder):
            print(f"Warning: Patient folder {input_folder} does not exist, skipped")
            continue

        # Define file paths
        t1_skull_path = os.path.join(output_folder, "T1_dwi_skull.nii.gz")  # Output from step 3
        t1_fmri_path = os.path.join(input_folder, "T1_fmri.nii.gz")  # T1WI template in fMRI space
        highfa_mask_path = os.path.join(output_folder, "edema_highFA_mask.nii.gz")  # Output from step 2
        lowfa_mask_path = os.path.join(output_folder, "edema_lowFA_mask.nii.gz")  # Output from step 2

        # Check required files
        required_files = [t1_skull_path, t1_fmri_path, highfa_mask_path, lowfa_mask_path]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"Warning: Missing files {missing_files}, skipped")
            continue

        try:
            # Load images for registration
            fixed_image = ants.image_read(t1_fmri_path)  # T1WI template (fMRI space)
            moving_image = ants.image_read(t1_skull_path)  # Skull-stripped T1 (DWI space)

            # Load masks
            highfa_mask = ants.image_read(highfa_mask_path)
            lowfa_mask = ants.image_read(lowfa_mask_path)

            # Perform SyN registration (optimal for brain imaging co-registration)
            print(f"Performing SyN registration for {patient_name}...")
            registration = ants.registration(
                fixed=fixed_image,
                moving=moving_image,
                type_of_transform='SyN',
                verbose=False
            )

            # Save registered T1 image
            registered_t1_path = os.path.join(output_folder, "T1_dwi_skull_registered_to_fmri.nii.gz")
            ants.image_write(registration['warpedmovout'], registered_t1_path)

            # Transform high-FA edema mask to fMRI space
            transformed_highfa = ants.apply_transforms(
                fixed=fixed_image,
                moving=highfa_mask,
                transformlist=registration['fwdtransforms'],
                interpolator="nearestNeighbor"  # Nearest neighbor for mask preservation
            )
            highfa_out_path = os.path.join(output_folder, "edema_highFA_mask_fmri_space.nii.gz")
            ants.image_write(transformed_highfa, highfa_out_path)

            # Transform low-FA edema mask to fMRI space
            transformed_lowfa = ants.apply_transforms(
                fixed=fixed_image,
                moving=lowfa_mask,
                transformlist=registration['fwdtransforms'],
                interpolator="nearestNeighbor"
            )
            lowfa_out_path = os.path.join(output_folder, "edema_lowFA_mask_fmri_space.nii.gz")
            ants.image_write(transformed_lowfa, lowfa_out_path)

            print(f"Completed registration for {patient_name}")
            print(f"Registered T1: {registered_t1_path}")
            print(f"Transformed high-FA mask: {highfa_out_path}")
            print(f"Transformed low-FA mask: {lowfa_out_path}")

        except Exception as e:
            print(f"Error processing {patient_name}: {str(e)}, skipped")


# -------------------------- 5. Generate 4-Region Atlas --------------------------
def generate_4region_atlas():
    """
    Generate 4-region atlas (tumor, edema_lowFA, edema_highFA, normal brain tissue)
    based on registered masks and T1WI template
    """
    print("\n=== Step 5: Generating 4-Region Atlas ===")
    processed = 0
    skipped = 0

    for patient_name in TEST_PATIENTS:
        input_folder = os.path.join(INPUT_ROOT_DIR, patient_name)
        output_folder = os.path.join(OUTPUT_ROOT_DIR, patient_name)
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\nProcessing patient: {patient_name}")
        
        if not os.path.exists(input_folder):
            print(f"Warning: Patient folder {input_folder} does not exist, skipped")
            skipped += 1
            continue

        # Define file paths
        t1_fmri_path = os.path.join(input_folder, "T1_fmri.nii.gz")  # T1WI template
        tumor_mask_path = os.path.join(input_folder, "T1seg_tumor_pair_fmri_space.nii.gz")
        lowfa_mask_path = os.path.join(output_folder, "edema_lowFA_mask_fmri_space.nii.gz")  # Output from step 4
        highfa_mask_path = os.path.join(output_folder, "edema_highFA_mask_fmri_space.nii.gz")  # Output from step 4

        # Check required files
        missing_files = []
        if not os.path.exists(t1_fmri_path):
            missing_files.append("T1_fmri.nii.gz")
        if not os.path.exists(tumor_mask_path):
            missing_files.append("T1seg_tumor_pair_fmri_space.nii.gz")
        if not os.path.exists(lowfa_mask_path):
            missing_files.append("edema_lowFA_mask_fmri_space.nii.gz")
        if not os.path.exists(highfa_mask_path):
            missing_files.append("edema_highFA_mask_fmri_space.nii.gz")

        if missing_files:
            print(f"Warning: Missing files {missing_files}, skipped")
            skipped += 1
            continue

        try:
            # Load image data
            t1_img = nib.load(t1_fmri_path)
            tumor_mask_img = nib.load(tumor_mask_path)
            lowfa_mask_img = nib.load(lowfa_mask_path)
            highfa_mask_img = nib.load(highfa_mask_path)

            t1_data = t1_img.get_fdata()
            tumor_mask = tumor_mask_img.get_fdata()
            lowfa_mask = lowfa_mask_img.get_fdata()
            highfa_mask = highfa_mask_img.get_fdata()
            affine = t1_img.affine
            header = t1_img.header

            # Generate normal brain mask (T1 valid region - tumor - edema)
            t1_mask = np.where(t1_data > 0, 1, 0)
            tumor_binary = np.where(tumor_mask > 0, 1, 0)
            lowfa_binary = np.where(lowfa_mask > 0, 1, 0)
            highfa_binary = np.where(highfa_mask > 0, 1, 0)

            normal_mask = t1_mask - tumor_binary - lowfa_binary - highfa_binary
            normal_mask = np.where(normal_mask > 0, 1, 0)

            # Save normal brain mask
            normal_mask_img = nib.Nifti1Image(normal_mask, affine, header)
            normal_mask_path = os.path.join(output_folder, "normal_brain_mask.nii.gz")
            nib.save(normal_mask_img, normal_mask_path)

            # Generate 4-region atlas:
            # 1 = Tumor, 2 = Edema_lowFA, 3 = Edema_highFA, 4 = Normal brain
            atlas_data = np.zeros_like(t1_data, dtype=np.int16)
            atlas_data[tumor_binary == 1] = 1
            atlas_data[lowfa_binary == 1] = 2
            atlas_data[highfa_binary == 1] = 3
            atlas_data[normal_mask == 1] = 4

            # Save atlas file
            atlas_img = nib.Nifti1Image(atlas_data, affine, header)
            atlas_path = os.path.join(output_folder, "atlas_file_4regions.nii.gz")
            nib.save(atlas_img, atlas_path)

            print(f"Generated normal brain mask: {normal_mask_path}")
            print(f"Generated 4-region atlas: {atlas_path}")
            processed += 1

        except Exception as e:
            print(f"Error processing {patient_name}: {str(e)}, skipped")
            skipped += 1

    print(f"\nAtlas generation completed - Processed: {processed}, Skipped: {skipped}")


# -------------------------- 6. Calculate 4×4 Functional Connectivity Matrix --------------------------
def calculate_fc_matrix():
    """
    Calculate 4×4 functional connectivity matrix using fMRI data and 4-region atlas
    Extract pairwise functional similarity indices (FSI)
    """
    print("\n=== Step 6: Calculating 4×4 Functional Connectivity Matrix ===")
    REGION_LABELS = {
        1: 'tumor',
        2: 'edema_lowFA',
        3: 'edema_highFA',
        4: 'normal'
    }
    processed = 0
    skipped = 0

    for patient_name in TEST_PATIENTS:
        input_folder = os.path.join(INPUT_ROOT_DIR, patient_name)
        output_folder = os.path.join(OUTPUT_ROOT_DIR, patient_name)
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\nProcessing patient: {patient_name}")
        
        if not os.path.exists(input_folder):
            print(f"Warning: Patient folder {input_folder} does not exist, skipped")
            skipped += 1
            continue

        # Define file paths
        fmri_path = os.path.join(input_folder, "bold.nii.gz")
        atlas_path = os.path.join(output_folder, "atlas_file_4regions.nii.gz")  # Output from step 5
        confounds_path = os.path.join(input_folder, "confounds.tsv")

        # Check required files
        missing_files = []
        if not os.path.exists(fmri_path):
            missing_files.append("bold.nii.gz")
        if not os.path.exists(atlas_path):
            missing_files.append("atlas_file_4regions.nii.gz")

        if missing_files:
            print(f"Warning: Missing files {missing_files}, skipped")
            skipped += 1
            continue

        try:
            # Load fMRI and atlas data
            fmri_img = nib.load(fmri_path)
            atlas_img = nib.load(atlas_path)

            # Check spatial dimension matching
            if fmri_img.shape[:3] != atlas_img.shape[:3]:
                print(f"Error: fMRI and atlas spatial dimensions mismatch, skipped")
                skipped += 1
                continue

            # Load confounds (if available)
            confounds = None
            if os.path.exists(confounds_path):
                confounds = pd.read_csv(confounds_path, sep='\t').values
                print(f"Loaded confounds for {patient_name} (shape: {confounds.shape})")

            # Extract region-specific time series from fMRI
            masker = NiftiLabelsMasker(
                labels_img=atlas_path,
                standardize=True,
                memory='nilearn_cache',
                verbose=0
            )
            time_series = masker.fit_transform(fmri_path, confounds=confounds)
            print(f"Extracted time series shape: {time_series.shape} (timepoints × regions)")

            # Calculate functional connectivity (Pearson correlation)
            connectivity_measure = ConnectivityMeasure(kind='correlation')
            fc_matrix = connectivity_measure.fit_transform([time_series])[0]

            # Save time series and FC matrix
            ts_save_path = os.path.join(output_folder, "region_time_series.npy")
            fc_matrix_path = os.path.join(output_folder, "fc_matrix_4x4.npy")
            fc_txt_path = os.path.join(output_folder, "fc_matrix_4x4.txt")

            np.save(ts_save_path, time_series)
            np.save(fc_matrix_path, fc_matrix)
            np.savetxt(fc_txt_path, fc_matrix, fmt='%.4f', 
                       header='\t'.join(REGION_LABELS.values()), 
                       comments='')

            # Print FC matrix summary
            print(f"4×4 FC matrix for {patient_name}:")
            print(fc_matrix.round(4))
            print(f"Saved time series: {ts_save_path}")
            print(f"Saved FC matrix: {fc_matrix_path}")

            processed += 1

        except Exception as e:
            print(f"Error processing {patient_name}: {str(e)}, skipped")
            skipped += 1

    print(f"\nFC matrix calculation completed - Processed: {processed}, Skipped: {skipped}")
    

# -------------------------- Step 7: Flatten FC Matrix into 10 Features (FIXED for header TXT) --------------------------
def flatten_fc_features():
    print("\n=== Step 7: Flattening 4×4 FC matrix into 10 features per patient ===")
    regions = ["tumor", "edemaLowFA", "edemaHighFA", "normal"]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    all_features = []

    for patient in TEST_PATIENTS:
        patient_dir = os.path.join(OUTPUT_ROOT_DIR, patient)
        matrix_path = os.path.join(patient_dir, "fc_matrix_4x4.txt")
        if not os.path.exists(matrix_path):
            print(f"Warning: No matrix found for {patient}, skipped")
            continue
        
        try:
            # ✅ 关键修复：读取所有行，并跳过以字母开头的标题行
            with open(matrix_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            # 只保留纯数字行（跳过第一行标题：tumor edema_lowFA ...）
            valid_lines = []
            for line in lines:
                # 如果这一行包含字母，说明是标题，跳过
                if any(c.isalpha() for c in line):
                    continue
                valid_lines.append(line)

            # 转换为矩阵
            matrix = []
            for line in valid_lines:
                row = list(map(float, line.split()))
                matrix.append(row)
            matrix = np.array(matrix, dtype=np.float64)

            if matrix.shape != (4, 4):
                print(f"Invalid matrix shape for {patient}, got {matrix.shape}")
                continue

            # 提取 10 个特征
            features = {"PatientName": patient}
            for i, j in pairs:
                features[f"{regions[i]}_{regions[j]}"] = matrix[i, j]
            for idx, region in enumerate(regions):
                features[f"{region}_sum"] = matrix[idx, :].sum() - 1.0

            all_features.append(features)
            print(f"Processed: {patient}")

        except Exception as e:
            print(f"Error processing {patient}: {str(e)}")
            continue

    if all_features:
        col_order = ["PatientName"] +                     [f"{regions[i]}_{regions[j]}" for i, j in pairs] +                     [f"{r}_sum" for r in regions]
        df = pd.DataFrame(all_features)[col_order]
        output_csv = os.path.join(OUTPUT_ROOT_DIR, "functional_connectivity_features.csv")
        df.to_csv(output_csv, index=False)
        print(f"\nAll 10 FC features saved to: {output_csv}")
        print(f"Total patients processed: {len(all_features)}")
    else:
        print("\nNo features extracted.")



# -------------------------- Main Execution --------------------------
if __name__ == "__main__":
    # Execute all steps in sequence
    calculate_fa_statistics()
    segment_edema_by_fa()
    skull_strip_t1_dwi()
    register_dwi_to_fmri()
    generate_4region_atlas()
    calculate_fc_matrix()
    flatten_fc_features()

    print("\n=== All FC feature engineering steps completed ===")
    print(f"All outputs saved to: {OUTPUT_ROOT_DIR}")

