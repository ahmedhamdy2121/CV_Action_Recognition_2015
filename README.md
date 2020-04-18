## Human Action Recognition by Representing 3D Skeletons as Points in a Lie

My additions to the paper can be found on the report below.

Report: https://1drv.ms/b/s!AjpxhCM4OMFHg_AOZkj75ky1ubXCjQ

Please cite the following paper if you use this code or data.

************************************************************************************

Raviteja Vemulapalli, Felipe Arrate, and Rama Chellappa, "Human Action Recognition 
by Representing 3D Human Skeletons as Points in a Lie Group", CVPR, 2014.

************************************************************************************


This code has been tested on a 64-bit Linux(Ubuntu 12.04) machine using MATLAB-R2012b.
If you are planning to run it on a Windows machine, make sure that you mex the 'dpcore.c' file in the folder "Code/DTW".
If you are running on windows 32 bits, then make the libsvm "read the README file in Code/libsvm-3.12/matlab"

Difference from 32 to 64:
1. DTW folder
2. svm folder
3. RF folder

Experimental setting:
Cross-subject - half of the subjects used for training and the remaining half used for testing.
Results are averaged over 10 different training and test subject combinations.


The matlab file "run.m" runs the experiments for UTKinect-Action, Florence3D-Action and MSRAction3D 
datasets using 5 different skeletal representations: 'JP', 'RJP', 'JA', 'BPL' and 'Proposed'.


The file "skeletal_action_classification.m" contains the code for entire pipeline:
Step 1: Skeletal representation ('JP' or 'RJP' or 'JA' or 'BPL' or 'Proposed')
Step 2: Temporal modeling (DTW and Fourier Temporal Pyramid)
Step 3: Classification: One-vs-All linear SVM (implemented as kernel SVM with linear kernel)

Note: It takes around 6 hrs to run the experiments for UTKinect and Florence3D datasets.
For MSRAction3D it may take a lot of time (DTW part is slow due to more action categories).
