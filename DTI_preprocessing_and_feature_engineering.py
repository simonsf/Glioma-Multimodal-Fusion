#!/usr/bin/env python
# coding: utf-8

# In[4]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DTI Preprocessing and Structural Similarity Index (SSI) Calculation
For SCI Code Availability: Glioma-Multimodal-Fusion

Functions:
1. Spatial co-registration of T1WI to DWI space
2. ROI mask transformation to DWI space
3. Region-specific DTI parameter extraction (FA, RA, AD, RD, MD)
4. Pairwise Structural Similarity Index (SSI) calculation

GitHub: https://github.com/simonsf/Glioma-Multimodal-Fusion
"""

import os
import ants
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 🔴 USER CONFIGURATION (DO NOT CHANGE OTHER PARTS)
# ==============================================================================
# ORIGINAL DATA DIRECTORY (EXACT PATH FROM YOUR ORIGINAL CODE)
INPUT_MAIN_DIR = ".../input/brain_mask/"

# OUTPUT DIRECTORY (SEPARATE, NO OVERWRITE TO ORIGINAL DATA)
OUTPUT_DIR = ".../results/DTI_output/"

# ONLY PROCESS THESE 3 PATIENTS FOR TESTING
TEST_PATIENTS = ['Patient01', 'Patient02', 'Patient03']
# ==============================================================================

# Automatically create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)


def register_and_transform():
    """
    Spatial co-registration and mask transformation.
    All outputs saved to OUTPUT_DIR.
    """
    for patient in TEST_PATIENTS:
        patient_dir = os.path.join(INPUT_MAIN_DIR, patient)
        if not os.path.isdir(patient_dir):
            print(f"Patient {patient} not found, skipped.")
            continue

        print(f"Processing patient: {patient}")

        t1_path = os.path.join(patient_dir, "T1.nii.gz")
        t1_dwi_path = os.path.join(patient_dir, "T1_dwi.nii.gz")
        tumor_mask_path = os.path.join(patient_dir, "T1_tumor_pair.nii.gz")
        edema_mask_path = os.path.join(patient_dir, "T1_edema.nii.gz")

        required_files = [t1_path, t1_dwi_path, tumor_mask_path, edema_mask_path]
        if not all(os.path.exists(f) for f in required_files):
            print(f"Missing files for {patient}, skipped.")
            continue

        try:
            fixed_image = ants.image_read(t1_dwi_path)
            moving_image = ants.image_read(t1_path)
            tumor_mask = ants.image_read(tumor_mask_path)
            edema_mask = ants.image_read(edema_mask_path)

            registration = ants.registration(
                fixed=fixed_image, moving=moving_image,
                type_of_transform='SyN', verbose=False
            )

            # Create patient output folder
            patient_out_dir = os.path.join(OUTPUT_DIR, patient)
            os.makedirs(patient_out_dir, exist_ok=True)

            # Transform tumor mask
            transformed_tumor = ants.apply_transforms(
                fixed=fixed_image, moving=tumor_mask,
                transformlist=registration['fwdtransforms'],
                interpolator="nearestNeighbor"
            )
            tumor_out = os.path.join(patient_out_dir, "T1_tumor_pair_dwi_space.nii.gz")
            ants.image_write(transformed_tumor, tumor_out)

            # Transform edema mask
            transformed_edema = ants.apply_transforms(
                fixed=fixed_image, moving=edema_mask,
                transformlist=registration['fwdtransforms'],
                interpolator="nearestNeighbor"
            )
            edema_out = os.path.join(patient_out_dir, "T1_edema_dwi_space.nii.gz")
            ants.image_write(transformed_edema, edema_out)

            print(f"Patient {patient}: registration done.")

        except Exception as e:
            print(f"Patient {patient} error: {str(e)}")


def calculate_dwi_parameters():
    """
    Extract mean DTI parameters for 3 regions: tumor, edema, normal.
    CSV saved to OUTPUT_DIR.
    """
    dwi_params = {'fa': 'FA', 'ra': 'RA', 'ad': 'AD', 'rd': 'RD', 'md': 'MD'}
    all_results = []

    for patient in TEST_PATIENTS:
        patient_dir = os.path.join(INPUT_MAIN_DIR, patient)
        patient_out_dir = os.path.join(OUTPUT_DIR, patient)

        tumor_mask_path = os.path.join(patient_out_dir, "T1_tumor_pair_dwi_space.nii.gz")
        edema_mask_path = os.path.join(patient_out_dir, "T1_edema_dwi_space.nii.gz")
        brain_mask_path = os.path.join(patient_dir, "brain_mask_dwi.nii.gz")

        if not all(os.path.exists(p) for p in [tumor_mask_path, edema_mask_path, brain_mask_path]):
            print(f"Missing masks for {patient}, skipped.")
            continue

        try:
            tumor_mask = ants.image_read(tumor_mask_path).numpy() > 0
            edema_mask = ants.image_read(edema_mask_path).numpy() > 0
            brain_mask = ants.image_read(brain_mask_path).numpy() > 0
            normal_mask = brain_mask & ~(tumor_mask | edema_mask)

            patient_results = {'PatientName': patient}

            for param, full_name in dwi_params.items():
                param_path = os.path.join(patient_dir, f"{param}.nii.gz")
                if not os.path.exists(param_path):
                    continue

                img = ants.image_read(param_path).numpy()
                patient_results[f"{full_name}_tumor"] = np.mean(img[tumor_mask]) if np.any(tumor_mask) else np.nan
                patient_results[f"{full_name}_edema"] = np.mean(img[edema_mask]) if np.any(edema_mask) else np.nan
                patient_results[f"{full_name}_normal"] = np.mean(img[normal_mask]) if np.any(normal_mask) else np.nan

            all_results.append(patient_results)
            print(f"Patient {patient}: parameters extracted.")

        except Exception as e:
            print(f"Patient {patient} error: {str(e)}")

    csv_out = os.path.join(OUTPUT_DIR, "dwi_parameter_results.csv")
    pd.DataFrame(all_results).to_csv(csv_out, index=False)
    return csv_out


def compute_ssi(csv_path):
    """
    Compute pairwise Structural Similarity Index (SSI) between regions.
    Results saved to OUTPUT_DIR.
    """
    if not os.path.exists(csv_path):
        print("CSV not found.")
        return

    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ["Label", "PatientName"]]

    for col in feature_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    params = ['FA', 'RA', 'AD', 'RD', 'MD']
    tumor_cols = [f'{p}_tumor' for p in params]
    edema_cols = [f'{p}_edema' for p in params]
    normal_cols = [f'{p}_normal' for p in params]

    results = []
    for _, row in df.iterrows():
        try:
            t = np.array([row[c] for c in tumor_cols])
            e = np.array([row[c] for c in edema_cols])
            n = np.array([row[c] for c in normal_cols])

            def corr(a, b):
                if np.all(a == a[0]) or np.all(b == b[0]):
                    return np.nan
                return np.corrcoef(a, b)[0, 1]

            results.append({
                "PatientName": row["PatientName"],
                "tumor_edema_corr": corr(t, e),
                "tumor_normal_corr": corr(t, n),
                "edema_normal_corr": corr(e, n)
            })
        except:
            continue

    ssi_df = pd.DataFrame(results)
    ssi_out = os.path.join(OUTPUT_DIR, "dwi_parameter_results_SSI.csv")
    ssi_df.to_csv(ssi_out, index=False)
    print("SSI calculation completed.")


if __name__ == "__main__":
    print("=== DTI Processing for SCI Test (3 Patients Only) ===")
    print(f"Input path: {INPUT_MAIN_DIR}")
    print(f"Output path: {OUTPUT_DIR}")
    print(f"Test patients: {TEST_PATIENTS}\n")

    register_and_transform()
    csv_file = calculate_dwi_parameters()
    compute_ssi(csv_file)

    print("\n✅ All tasks finished! Results saved to separate output directory.")


# In[ ]:




