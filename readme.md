```python
# Glioma Multimodal Imaging Fusion: Structural & Functional Feature Analysis

This repository contains the custom Python code for multimodal magnetic resonance imaging (MRI) analysis of glioma. The code implements core preprocessing, region-of-interest (ROI) feature extraction, and quantitative index calculation for diffusion tensor imaging (DTI) and functional MRI (fMRI) data.

## Code Availability
All custom code in this repository is released under an open-source license and is fully accessible at:  
https://github.com/simonsf/Glioma-Multimodal-Fusion

Critical custom analytical steps implemented in this code include:
- Spatial co-registration of DTI and fMRI maps to T1-weighted imaging (T1WI) templates
- Extraction of region-specific parameters across tumor, edema, and normal brain masks
- Calculation of pairwise structural similarity index (SSI) and functional similarity index (FSI)
- Functional connectivity (FC) model feature engineering

## Core Code Files
This repository includes three key Python scripts for multimodal imaging processing:

### 1. `DTI_preprocessing_and_feature_engineering.py`
Processes diffusion tensor imaging (DTI) data for structural brain analysis:
- T1WI registration to DWI space and mask transformation
- Extraction of 5 key DTI parameters (FA, RA, AD, RD, MD)
- Calculation of Structural Similarity Index (SSI) across tumor, edema, and normal brain regions
- Automated batch processing with structured output files
    
### 2. `fMRI_preprocessing_and_feature_engineering.py`
Processes functional MRI (fMRI) data for glioma analysis:
- T1WI skull stripping and spatial registration to fMRI space
- ROI feature extraction for 5 core fMRI metrics (SmKccReHo, SmCoHeReHo, mALFF, mfALFF, mPerAF)
- Calculation of Functional Similarity Index (FSI) across tumor, edema, and normal brain regions
- Batch processing and CSV output of ROI features and FSI results

### 3. `FC_preprocessing_and_feature_engineering.py`
Performs advanced functional connectivity (FC) analysis:
- FA-based edema segmentation (low-FA / high-FA subregions)
- 4-region atlas generation (tumor, low-FA edema, high-FA edema, normal brain)
- 4×4 FC matrix construction from fMRI BOLD signals
- FC feature flattening and standardization for predictive modeling

## Dependencies
The code relies on standard neuroimaging and data science Python libraries:
- `nibabel` (NIfTI image I/O)
- `ANTsPy` (image registration)
- `nilearn` (functional connectivity analysis)
- `scikit-learn` (feature standardization)
- `numpy` / `pandas` (numerical processing & CSV output)

## Usage
All scripts are designed for batch processing of glioma MRI data. Configure input/output paths and patient lists at the top of each script, then run directly:
```bash
python fMRI_preprocessing_and_feature_engineering.py
python DTI_preprocessing_and_feature_engineering.py
python FC_preprocessing_and_feature_engineering.py
```
