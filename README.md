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

## Calculation of Stokes parameters
We compute the Stokes parameters as follows. ${S}_{0}^{\prime} $ can be obtained as:
$$
\begin{align}  
	{S}_{0}^{\prime} & = \frac{1}{2}\left( {{S}_{0}} + \cos 2\varphi \cos \left( 2\theta - 2\varphi \right) {S}_{1} \right. \notag \\ 
	&\phantom{{}={}} \left. + \sin 2\varphi \cos \left( 2\theta - 2\varphi \right) {{S}_{2}} + \sin \left( 2\theta - 2\varphi \right) {{S}_{3}} \right) 
\end{align}
$$

With a fixed linear polarizer angle $ \theta $, rotate the quarter-waveplate to angles $ {{\varphi }_{1}},{{\varphi }_{2}},\ldots {{\varphi }_{n}} $, and collect the corresponding images. The results can be obtained as follows:
$$
\begin{align}
	& \left( \begin{matrix}
		{{S}_0^{\prime}}\left( \theta ,{{\varphi }_{1}} \right)  \\
		{{S}_0^{\prime}}\left( \theta ,{{\varphi }_{2}} \right)  \\
		\vdots   \\
		{{S}_0^{\prime}}\left( \theta ,{{\varphi }_{n}} \right)  \\
	\end{matrix} \right)=\frac{1}{2}\left( \begin{matrix}
		1 & \cos 2{{\varphi }_{1}} \cos \left( 2\theta -2{{\varphi }_{1}} \right)  \\
		1 & \cos 2{{\varphi }_{2}} \cos \left( 2\theta -2{{\varphi }_{2}} \right)  \\
		\vdots  & \vdots   \\
		1 & \cos 2{{\varphi }_{n}} \cos \left( 2\theta -2{{\varphi }_{n}} \right)  \\
	\end{matrix} \right.  \notag \\ 
	& \quad \quad \quad \left. \begin{matrix}
		\sin 2{{\varphi }_{1}} \cos \left( 2\theta -2{{\varphi }_{1}} \right) & \sin \left( 2\theta -2{{\varphi }_{1}} \right)  \\
		\sin 2{{\varphi }_{2}} \cos \left( 2\theta -2{{\varphi }_{2}} \right) & \sin \left( 2\theta -2{{\varphi }_{2}} \right)  \\
		\vdots  & \vdots   \\
		\sin 2{{\varphi }_{n}} \cos \left( 2\theta -2{{\varphi }_{n}} \right) & \sin \left( 2\theta -2{{\varphi }_{n}} \right)  \\
	\end{matrix} \right)\left( \begin{matrix}
		{{S}_{0}}  \\
		{{S}_{1}}  \\
		{{S}_{2}}  \\
		{{S}_{3}}  \\
	\end{matrix} \right)
\end{align}
$$

Eq.~\ref{eq:matrixS0'} can be simplified as:
$$
\begin{equation}
	\mathbf{S}_0^{\prime}=\mathbf{A}\cdot \mathbf{S}
\end{equation}
$$

When $\mathbf{A}$ is non-ill-conditioned and its rank is greater than or equal to 4, the least squares method can be used to obtain the optimal approximate solution to the above contradictory system of equations:
$$
\begin{equation}
	\mathbf{S}={{\left( {{\mathbf{A}}^{\text{T}}}\cdot \mathbf{A} \right)}^{-1}}\cdot {{\mathbf{A}}^{\text{T}}}\cdot \mathbf{S}_0^{\prime} 
\end{equation}
$$

## Citation  
If our project supports your research, please cite our Applied Optics paper:  

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


