#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

"""
Custom code for fMRI preprocessing, region-specific feature extraction, and Functional Similarity Index (FSI) calculation
for glioma multimodal imaging analysis. This script implements:
1. Spatial co-registration of T1WI to fMRI parametric maps
2. Extraction of fMRI parameters across tumor/edema/normal brain masks
3. Calculation of pairwise Functional Similarity Indices (FSI)
"""

import os
import ants
import numpy as np
import pandas as pd
from glob import glob
from sklearn.preprocessing import StandardScaler
import nibabel as nib

# ========================== GLOBAL CONFIGURATION ==========================
# Original data paths (keep the exact paths from original code)
SEG_ROOT = ".../input/brain_mask/"
FMRI_ROOT = ".../input/RESTplus/"

# Independent output path (avoid polluting original data)
OUTPUT_ROOT = ".../results/fMRI_output/"

# Specify 3 test patients (replace with actual patient IDs from your dataset)
TEST_PATIENTS = ['Patient01', 'Patient02', 'Patient03']

# ANTs registration parameters for T1-fMRI co-registration (FULL PARAMS RETAINED)
REG_PARAMS = {
    'type_of_transform': 'SyN',
    'reg_iterations': (100, 50, 10),
    'flow_sigma': 3,
    'total_sigma': 0,
    'grad_step': 0.1,
    'verbose': False
}

# Define fMRI parameters and ROI regions for feature extraction
REGION_ORDER = ['tumor', 'edema', 'normal']
FSI_PARAMS = ['SmKccReHo', 'SmCoHeReHo', 'mALFF', 'mfALFF', 'mPerAF']  # ONLY THESE 5 PARAMS USED
# ========================== CORE FUNCTIONS ==========================

def skull_stripping_t1(patient_id):
    """
    Perform skull-stripping on T1_seg image using brain mask
    :param patient_id: Unique identifier for the patient
    :return: Path to skull-stripped T1 image or None if failed
    """
    t1_path = os.path.join(SEG_ROOT, patient_id, "T1_seg.nii.gz")
    mask_path = os.path.join(SEG_ROOT, patient_id, 'brain_mask_T1_seg.nii.gz')
    output_path = os.path.join(OUTPUT_ROOT, patient_id, "T1_seg_skull.nii.gz")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not all(os.path.exists(p) for p in [t1_path, mask_path]):
        print(f"[SKULL STRIP SKIP] Missing files for {patient_id}")
        return None
    
    try:
        # Load T1 and brain mask
        t1_img = nib.load(t1_path)
        mask_img = nib.load(mask_path)
        t1_data = t1_img.get_fdata()
        mask_data = mask_img.get_fdata()
        
        # Apply binary mask for skull-stripping
        mask_binary = np.where(mask_data > 0, 1, 0)
        t1_skull_data = t1_data * mask_binary
        
        # Save skull-stripped image
        t1_skull_img = nib.Nifti1Image(t1_skull_data, t1_img.affine, t1_img.header)
        nib.save(t1_skull_img, output_path)
        print(f"[SKULL STRIP DONE] {patient_id} -> {output_path}")
        return output_path
    except Exception as e:
        print(f"[SKULL STRIP ERROR] {patient_id}: {str(e)}")
        return None

def register_t1_to_fmri(patient_id):
    """
    Co-register skull-stripped T1WI to fMRI (KccReHo) space and transform tumor/edema masks
    :param patient_id: Unique identifier for the patient
    :return: Dictionary of ROI masks in fMRI space or None if failed
    """
    # Get file paths
    seg_dir = os.path.join(SEG_ROOT, patient_id)
    t1_skull_path = os.path.join(OUTPUT_ROOT, patient_id, "T1_seg_skull.nii.gz")
    tumor_mask_path = os.path.join(seg_dir, "T1_tumor_pair_T1seg_space.nii.gz")
    edema_mask_path = os.path.join(seg_dir, "T1_edema_T1seg_space.nii.gz")
    
    # Find fMRI reference image (KccReHo)
    fmri_ref_path = None
    for root, _, files in os.walk(FMRI_ROOT):
        if patient_id in root and "KccReHo.nii" in files:
            fmri_ref_path = os.path.join(root, "KccReHo.nii")
            break
    
    # Validate input files
    if not all(os.path.exists(p) for p in [t1_skull_path, tumor_mask_path, edema_mask_path, fmri_ref_path]):
        print(f"[REGISTRATION SKIP] Missing files for {patient_id}")
        return None
    
    try:
        # Load images for registration
        fixed_fmri = ants.image_read(fmri_ref_path)
        moving_t1 = ants.image_read(t1_skull_path)
        
        # Perform SyN co-registration (FULL PARAMS USED)
        reg_result = ants.registration(fixed=fixed_fmri, moving=moving_t1, **REG_PARAMS)
        
        # Save registered T1
        reg_t1_path = os.path.join(OUTPUT_ROOT, patient_id, "T1_reg_to_fmri.nii.gz")
        ants.image_write(reg_result['warpedmovout'], reg_t1_path)
        
        # Transform tumor/edema masks to fMRI space
        roi_masks = {}
        for mask_type, mask_path in [('tumor', tumor_mask_path), ('edema', edema_mask_path)]:
            mask = ants.image_read(mask_path)
            reg_mask = ants.apply_transforms(
                fixed=fixed_fmri,
                moving=mask,
                transformlist=reg_result['fwdtransforms'],
                interpolator='nearestNeighbor'
            )
            mask_output_path = os.path.join(OUTPUT_ROOT, patient_id, f"{mask_type}_reg_to_fmri.nii.gz")
            ants.image_write(reg_mask, mask_output_path)
            roi_masks[mask_type] = reg_mask.numpy() > 0.5
        
        # Create normal brain mask
        brain_mask = fixed_fmri.numpy() > 0
        roi_masks['normal'] = brain_mask & ~roi_masks['tumor'] & ~roi_masks['edema']
        
        print(f"[REGISTRATION DONE] {patient_id}")
        return {
            'fmri_ref': fmri_ref_path,
            'roi_masks': roi_masks,
            'patient_dir': os.path.dirname(fmri_ref_path)
        }
    except Exception as e:
        print(f"[REGISTRATION ERROR] {patient_id}: {str(e)}")
        return None

def extract_roi_features(patient_id, reg_results):
    """
    ONLY extract 5 core fMRI parameters for FSI
    """
    if not reg_results:
        return None
    
    try:
        roi_stats = {'patient': patient_id}
        fmri_files = glob(os.path.join(reg_results['patient_dir'], "*.nii"))
        
        for fmri_path in fmri_files:
            param_name = os.path.basename(fmri_path).split('.')[0]
            if param_name not in FSI_PARAMS:
                continue
            
            fmri_img = ants.image_read(fmri_path)
            fmri_data = fmri_img.numpy()
            
            for region in REGION_ORDER:
                mask = reg_results['roi_masks'][region]
                roi_mean = np.nanmean(fmri_data[mask]) if np.any(mask) else np.nan
                roi_stats[f"{param_name}_{region}_mean"] = roi_mean
        
        print(f"[FEATURE EXTRACTION DONE] {patient_id}")
        return roi_stats
    except Exception as e:
        print(f"[FEATURE EXTRACTION ERROR] {patient_id}: {str(e)}")
        return None

def sort_feature_columns(df):
    """
    Sort ONLY 5 core features (16 columns total)
    """
    ordered = ["patient"]
    for region in REGION_ORDER:
        for p in FSI_PARAMS:
            ordered.append(f"{p}_{region}_mean")
    return df[ordered]

def calculate_fsi(feature_df):
    """
    Calculate FSI using ONLY 5 core fMRI parameters
    """
    regions = {
        'tumor': [f'{p}_tumor_mean' for p in FSI_PARAMS],
        'edema': [f'{p}_edema_mean' for p in FSI_PARAMS],
        'normal': [f'{p}_normal_mean' for p in FSI_PARAMS]
    }

    for r, cols in regions.items():
        missing = [c for c in cols if c not in feature_df.columns]
        if missing:
            raise ValueError(f"Missing FSI features for {r}: {missing}")

    all_cols = regions['tumor'] + regions['edema'] + regions['normal']
    fsi_df = feature_df[["patient"] + all_cols].copy()

    for c in all_cols:
        if fsi_df[c].isnull().any():
            fsi_df[c].fillna(fsi_df[c].mean(), inplace=True)

    scaler = StandardScaler()
    fsi_df[all_cols] = scaler.fit_transform(fsi_df[all_cols])

    results = []
    for _, row in fsi_df.iterrows():
        pid = row["patient"]
        t = np.array([row[c] for c in regions['tumor']], dtype=np.float64)
        e = np.array([row[c] for c in regions['edema']], dtype=np.float64)
        n = np.array([row[c] for c in regions['normal']], dtype=np.float64)

        if len(t) != 5 or len(e) != 5 or len(n) != 5:
            continue

        def corr(a, b):
            if np.all(a == a[0]) or np.all(b == b[0]):
                return np.nan
            return np.corrcoef(a, b)[0, 1]

        results.append({
            "patient": pid,
            "tumor_edema_corr": corr(t, e),
            "tumor_normal_corr": corr(t, n),
            "edema_normal_corr": corr(e, n)
        })

    res_df = pd.DataFrame(results)
    print("\n=== FSI Summary ===")
    print(res_df[['tumor_edema_corr','tumor_normal_corr','edema_normal_corr']].describe())
    return res_df

# ========================== MAIN WORKFLOW ==========================
def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    print(f"[START] Processing test patients: {TEST_PATIENTS}")
    
    feature_list = []
    for patient in TEST_PATIENTS:
        skull_stripping_t1(patient)
        reg_results = register_t1_to_fmri(patient)
        roi_features = extract_roi_features(patient, reg_results)
        if roi_features:
            feature_list.append(roi_features)
    
    if not feature_list:
        print("[ERROR] No valid features extracted")
        return
    
    feature_df = pd.DataFrame(feature_list)
    sorted_df = sort_feature_columns(feature_df)
    
    feat_out = os.path.join(OUTPUT_ROOT, "fmri_roi_features.csv")
    sorted_df.to_csv(feat_out, index=False, encoding='utf-8-sig')
    print(f"\n[FEATURES SAVED] 16-column ROI features: {feat_out}")
    
    df_fsi = calculate_fsi(sorted_df)
    fsi_out = os.path.join(OUTPUT_ROOT, "fmri_fsi_results.csv")
    df_fsi.to_csv(fsi_out, index=False, encoding='utf-8-sig')
    print(f"[FSI SAVED] FSI results: {fsi_out}")
    
    print("\n[COMPLETED] All processing finished!")

if __name__ == "__main__":
    main()

