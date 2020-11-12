# LSCI - Laser Speckle Contrast Imaging Toolbox

## Summary

The LSCI Toolbox is a command line tool written in Matlab that can process raw LSCI image(s) data and extract contrast (K) map, tc )(correlation time) map, velocity (V) map and other useful data.

The LSCI toolbox contains series of scripts and functions that implement the main LASCA (Laser Speckle Contrast Analysis) methods - sLASCA (Spatial LASCA), stLASCA (Spatio-Temporal LASCA), tLASCA (Temporal LASCA), meLASCA (Multi-exposure LASCA). 

It also contains few other methods that might be of interest - tFDLSI (Temporal Frequency Domain Laser Speckle Imaging), teLASEA (Temporal Entropy Laser Speckle Entropy Analysis), fftLSA (Fourier Transform Laser Speckle Analysis).

It also contains some helper scripts/functions that facilitate pre- or post-processing steps, for example. converting a video to a multi-page tiff file (3D stack) for easy inspection with ImageJ or the like.

Below follows more information and explanation on using the toolbox.

## Introduction

The LSCI Toolbox was written by me for a scientific article in order to process LSCI raw data the respective experiment generated. This toolbox is part of the paper, therefore making it public. Also, I decided to share it with the public since at the time I could not find any similar software that can process raw LSCI data and extract contrast, correlation time and velocity maps. So this toolbox might be helpful for other people as well.

## Requirements

The toolbox was written and tested with Matlab 2019a (Windows 10, x64). 

You need to add the root directory and its subdirectories to your Matlab path in order to use the functions.

If you use big 3D stacks you will need a PC with enough RAM to load and process the stacks. RAM with 32 GB will allow to comfortably work with big stacks of images. A stack of 1000 images with size of 1000x1000 px and 16-bit depth with occupy in the RAM 2 GB. Note also that the toolbox produces output stacks for K map, tc map, V map, which are almost as big as the input stack size. Thus, the RAM usage will increase 4x, plus Matlab, the OS, the open applications will occupy additional space (few GB). Therefore, loading and processing big chunks of data might be a problem on low RAM PCs. In case of problems, to fit the data in the RAM, you can try to crop the stack to smaller size, i.e. only the ROI where you are interested to get the results.  

## LASCA Functions and Directory Structure

It follows short explanation of the directory structure and scripts/functions:

###### Root directory - contains all implemented LASCA methods (this are the methods that are meant to analyze LSCI raw image data):
1. lsci_sLASCA.m --> function that implements Spatial LASCA
2. lsci_stLASCA.m --> function that implements Spatio-Temporal LASCA
3. lsci_tLASCA.m --> function that implements Temporal LASCA
4. lsci_meLASCA.m --> function that implements Multi-exposure LASCA
5. lsci_tFDLSI.m --> function that implements Temporal Frequency Domain Laser Speckle Imaging analysis
6. lsci_teLASEA.m  --> function that implements (Temporal) Entropy LASEA (Entropy is used as an estimotor instead of Contrast)
7. lsci_fftLSA.m  --> function that uses Fourier transform to analize LSCI raw data and compare it with some LASCA methods

###### Subdirectories:
- 'data/' --> store data, mainly for sample files and/or other useful example data
- 'data/samples' --> sample LSCI data to play and test the toolbox
- 'io/' --> input-output helper functions, for example, to convert avi to multi-page tiff file (see 'io/lsci_VideosToMultiTiff.m' function)
- 'lib/' --> external ibraries or other similar libs
- 'lib/img/tiffwriter' --> fast tiff writer class (not used at the moment), but might be handy for some really big stacks, so I keep it just in case
- 'lib/sim/' --> mex functions to speed up simulation of laser speckle generation (not used by the LASCA functions)
- 'scripts/' --> some useful scripts for plotting results of LASCA analysis
- 'sim/' --> functions that simulate laser speckle (note: only 'sim/lsci_gnrLaserSpeckle3D.m' works correctly)
- 'system/' --> helper functions to interact with the system
- 'tools/' --> helper functions (not used by the LASCA methods) to analyze laser speckle statistics properties etc

## Usage - How It Works

All main LSACA methods implemented are in the root directory. The functions do not have GUI interface, so they must be run in the command window of Matlab. During run time some progress info will be printed/shown in the command line output to indicate what is being done, progress some statistics etc. 

In the 'data/samples/' directory there is some example of LSCI laser speckle image stack that can be used for initial play and tests. Note: most of the code is very well documented and commented so that it is relatively easy to understand what is going on and why inside the code.

Below are given examples how to use the LASCA functions. Before running a LASCA function go to the directory where the LSCI data is located.

###### Example 1: Usage of Spatial LASCA:

###### Example 2: Usage of Spatio-Temporal LASCA:

###### Example 3: Usage of Temporal LASCA:

###### Example 4: Usage of Multi-exposure LASCA:

###### Example 5: Usage of (Temporal) Frequency Domain Laser Speckle Imaging analysis:

###### Example 6: Usage of (Temporal) Entropy LASEA:

###### Example 7: Usage of Fourier Transform Laser Speckle Analysis:

## References and Literature

1. W. James Tom et al, "Efficient Processing of Laser Speckle Contrast Images", DOI: https://doi.org/10.1109/TMI.2008.925081
2. Julio C. Ramirez-San-Juan et al, "Impact of velocity distribution assumption on simplified laser speckle imaging equation ", DOI: https://doi.org/10.1364/OE.16.003197
3. P. Miao et al, "Random process estimator for laser speckle imaging of cerebral blood flow", DOI: https://doi.org/10.1364/OE.18.000218
4. Oliver Thomson et al, "Tissue perfusion measurements: multiple-exposure laser speckle analysis generates laser Doppler-like spectra", DOI: https://doi.org/10.1117/1.3400721
5. Hao Li et al, "Directly measuring absolute flow speed by frequency-domain laser speckle imaging", DOI: https://doi.org/10.1364/OE.22.021079
6. Tomi Smausz et al, "Real correlation time measurement in laser speckle contrast analysis using wide exposure time range images", DOI: https://doi.org/10.1364/AO.48.001425
7. Ashwin B. Parthasarathy et al, "Robust flow measurement with multi-exposure speckle imaging", DOI: https://doi.org/10.1364/OE.16.001975
8. Ping Kong et al, "A novel highly efficient algorithm for laser speckle imaging", DOI: https://doi.org/10.1016/j.ijleo.2016.04.004
9. S. Kazmi et al, "Evaluating multi-exposure speckle imaging estimates of absolute autocorrelation times", DOI: https://doi.org/10.1364/OL.40.003643
10. D. Boas et al, "Laser speckle contrast imaging in biomedical optics", DOI: https://doi.org/10.1117/1.3285504
11. Mitchell A. Davis et al, "Sensitivity of laser speckle contrast imaging to flow perturbations in the cortex", DOI: https://doi.org/10.1364/BOE.7.000759
12. Ashwin B. Parthasarathy et al, "Quantitative imaging of ischemic stroke through thinned skull in mice with Multi Exposure Speckle Imaging", DOI: https://doi.org/10.1364/BOE.1.000246
13. Annemarie Nadort et al, "Quantitative laser speckle flowmetry of the in vivo microcirculation using sidestream dark field microscopy", DOI: https://doi.org/10.1364/BOE.4.002347
14. Peng Miao et al, "Entropy analysis reveals a simple linear relation", DOI: http://dx.doi.org/10.1364/OL.39.003907
15. Juan A Bonachela et al, "Entropy estimates of small data sets", DOI: http://dx.doi.org/10.1088/1751-8113/41/20/202001
 
