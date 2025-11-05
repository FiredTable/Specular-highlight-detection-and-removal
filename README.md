# Specular Highlight Detection and Removal  
**Full-Polarization Image Dataset for Highly Reflective Workpieces**  

This repository provides a high-quality dataset of polarization images designed to detect and remove **specular highlights** from highly reflective industrial workpieces under complex illumination, enabling robust surface defect inspection.  

## Dataset Overview  
We introduce a novel polarization imaging dataset collected using the **quarter-waveplate rotation method**. Designed to address challenges in industrial inspection, this dataset focuses on capturing **multidirectional polarization states** from workpieces exhibiting strong specular reflections and subtle defects (e.g., scratches, pits) under mixed natural and polarized lighting.  

Key features:  
🔶 **Diverse Industrial Scenes**: 32 distinct industrial settings with complex reflection patterns  
🔶 **Robust Illumination**: Simulates challenging mixed-light environments (natural + polarized)  
🔶 **Defect Emphasis**: Surface defects remain visible despite strong specular highlights  
🔶 **High-Resolution**: All images captured at 1224 × 1024 resolution  

![Sample Polarization Images](images/examples_of_FPID.png)  
*Example polarization images showing specular highlights and surface defects*  

## Acquisition Methodology  
Data was captured using a custom polarization imaging system featuring:  
- **Rotation Mechanism**: Quarter-waveplate rotated in 10° increments (0°→180°)  
- **Fixed Polarizer**: Linear polarizer at stationary orientation  
- **Image Collection**: 19 polarization images per rotation sequence  
- **Total Volume**: **608 images** (32 scenes × 19 rotations)  

This methodology enables comprehensive multi-angle polarization analysis while preserving defect visibility in high-reflectance environments.  

## Access Dataset  
The full dataset is publicly available for research purposes:  
**📥 Download Link**: https://pan.quark.cn/s/69ad85896146 

> ⚠️ **Note**: Simply rotating the quarter-waveplate proves insufficient for complete specular highlight removal in mixed-light industrial setups. This dataset specifically enables development of advanced highlight-suppression techniques.  

## Method Workflow
Our proposed approach follows a systematic pipeline for highlight detection and removal:

![Method Flow](images/graphicalabstract.jpg)  
*Fig 2: Four-stage workflow:  
① Full-polarization image acquisition →  
② Polarization feature analysis →  
③ Specular detection via feature clustering →  
④ Highlight suppression using reflection masks*

## Citation  
If this dataset supports your research, please cite our Applied Optics paper:  

```bibtex
@article{zhou_shdr2025,
  author  = {...},  
  title   = {...},  
  journal = {Applied Optics},
  volume  = {...},
  year    = {2025},
  pages   = {...},
  doi     = {...}    % Waiting for publication
}


