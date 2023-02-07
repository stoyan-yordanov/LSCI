# LSCI - Laser Speckle Contrast Imaging Toolbox

## Summary

The LSCI Toolbox is a command line tool written in Matlab that can process raw LSCI image(s) and extract contrast K map, correlation time tc map, velocity V map and other useful data.

The LSCI toolbox contains series of scripts and functions that implement the main LASCA (Laser Speckle Contrast Analysis) methods - sLASCA (Spatial LASCA), stLASCA (Spatio-Temporal LASCA), tLASCA (Temporal LASCA), meLASCA (Multi-exposure LASCA). 

It also contains few other methods that might be of interest - tFDLSI (Temporal Frequency Domain Laser Speckle Imaging), teLASEA (Temporal Entropy Laser Speckle Entropy Analysis), fftLSA (Fourier Transform Laser Speckle Analysis).

It also contains some helper scripts/functions that facilitate pre- or post-processing steps. For example, converting a video to a multi-page tiff file (3D stack) for easy inspection with ImageJ or the like.

Initially the LSCI toolbox was created and used for processing the data in the paper: "Real-time monitoring of biomechanical activity in aphids by laser speckle contrast imaging" (Optics Express Vol. 29, Issue 18, pp. 28461-28480 (2021), DOI: https://doi.org/10.1364/OE.431989). The main reason for creating and later making the code publically available is the lack of such existing code that can do the respective LSCI data processing. So I decided to write such tool and make it available for everybody intrested or needing to process LSCI data.

Below follows more information and explanation on using the toolbox. 

## Introduction

The LSCI Toolbox was written by me for a scientific article in order to process LSCI raw data that the respective experiments generated. This toolbox is part of the paper, therefore making it public. Also, I decided to share it with the public since at the time I could not find any similar software that can process raw LSCI data and extract contrast, correlation time and velocity maps. So this toolbox might be helpful for other people as well.

## Requirements

Download the repository to your PC (use the green button in the upper right). Unzip to desired directory. You need to add the root directory and its subdirectories to your Matlab path in order to use the functions.

The toolbox was written and tested with Matlab 2019a (Windows 10, x64). To run the functions in the toolbox you must have the following Matlab Toolboxes installed:
- Optimization Toolbox
- Image Processing Toolbox
- Curve Fitting Toolbox
- Signal Processing Toolbox

If you use big 3D stacks you will need a PC with enough RAM to load and process the stacks. A PC with 32 GB RAM will allow to comfortably work with big stacks of images. A stack of 1000 images with the size of 1000x1000 px and 16-bit depth will occupy in the 2 GB in the RAM. Note also that the toolbox produces output stacks for K map, tc map, V map, which are almost as big as the input stack size. Thus, the RAM usage will increase 4x, plus add to this Matlab itself, the OS, the open applications and it will occupy additional space (few more GB). Therefore, loading and processing big chunks of data might be a problem on low RAM PCs. In case of problems, to fit the data in the RAM, you can try to crop the stack to smaller size, i.e. only the ROI where you are interested to get the results.  

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
- 'tools/' --> helper functions (not used by the LASCA methods) to analyze laser speckle statistical properties etc

## Usage - How It Works

All main LSACA methods implemented are in the root directory. The functions do not have GUI interface, so they must be run in the command window of Matlab. During run time some progress info will be printed/shown in the command line output to indicate what is being done, progress some statistics etc. 

In the 'data/samples/' directory there is some example of LSCI laser speckle image stack that can be used for initial play and tests. Note: most of the code is very well documented and commented so that it is relatively easy to understand what is going on and why inside the code.

Below are given examples how to use the LASCA functions. Before running a LASCA function go to the directory where the LSCI data is located.

Depending on the LASCA algorithm one uses there could be a decrease in spatial and/or temporal resolution. For example, the three major algorithms - sLASCA (Spatial LASCA), tsLASCA (Temporal LASCA), and stLASCA (Spatio-Temporal LASCA) make different use of the available pixels and frames. 

The sLASCA uses a 2D pixel window (e.g. 5x5 = 25 pixels) to calculate the speckle contrast K, which means it assigns a single K value to a group of pixels (using their intensities), which is basically kind of binning all pixel in the given window to get one speckle contrast value K. When we have such calculation of K we lose spatial resolution for the K map. This means the output K image will be with decreased resolution compare the input intensity image.

The tLASCA is similar to sLASCA, but in time - i.e. we lose temporal resolution to calculate K, but we preserve the spatial resolution of the K map.

The stLASCA is a combination between sLASCA and tLASCA, so you lose spatial and temporal resolution at the same time, but less than the pure spatial or temporal LASCA algorithms.

Further notes:

The choice of the exposure time value in sLASCA/tLASCA/stLASCA might be improtant to obtain good results. According to my experience and to decrease the error in estimating K the best is to have long exposure times so that the resulting contrast value K is below < 0.5, ideally even below < 0.1. If the contrast K is too high it means nothing changed in the speckle and then calculating K statistically is with very big uncertainty (one might even get K > 1). So values of K > 0.5 would be a clear sign that you need longer exposure time to obtain K with good statistical accuracy. Therefore, choose an exposure time such that you see big variations in speckle pattern from one frame to another (this is the rule of thumb).

Also in the paper above related to the LSCI Toolbox I explain in detail the theory and quirks of the LSCI so that one can get a better idea how to use the toolbox. The detailed explanation that is worth reading is in the [Supporting Information](https://opticapublishing.figshare.com/articles/journal_contribution/Supplementary_document_for_Real-time_monitoring_of_biomechanical_activity_in_aphids_by_laser_speckle_contrast_imaging_-_5356065_pdf/15023469) - I strongly recommend reading it, because it is like a review and tutorial of LSCI for newbies, and I wrote it for myself, otherwise the information about LSCI is scattered in many papers. I am an optical microscopy expert, but not an expert in LSCI - I had to use it in one project that resulted in a paper related mentioned above, and decided to put all important LSCI theory in one document (the SI) for future references if needed. This is why it will be useful for everybody starting with LSCI, and this is why I wrote the LSCI Toolbox, because at that time I could not find anything useful as available LSCI software tools.

###### Example 1: Usage of Spatial LASCA:

To process LSCI raw data stack by sLASCA type and run in the command prompt of Matlab the following function (choose a file to process by typing the respective number when asked, then press Enter):

```matlab
>> lsci_sLASCA('', 1, 100, 'ConvFilter', [200, 200, 1], 5, 4e-3, 0.633, 0.1, 4) 
```

Explanation of the arguments of the function:
- arg 1: InputFile = '' --> empty will cause the function to ask for a file to process
- arg 2: StartFrame = 1 --> start frame of the processing (in this case first frame, but can start from a different frame)
- arg 3: EndFrame = 100 --> end frame to process (processes all frames between StartFrame and EndFrame)
- arg 4: NumericalMethod = 'ConvFilter' --> how to calc the contrast K, available methods from slowest to fastest: 'Direct' (slowest) | 'Sums' | 'SumsVec' | 'ConvFilter' (fastest)
- arg 5: PixelXYZ = [200, 200, 1] --> [X, Y, Z] array to specify coordinate of the pixel/point where we show statistics for K (contrast) and V (velocity) in the command line report
- arg 6: XYWindowSizePx = 5 --> means 5x5 pixel window, i.e. pixel size of the XY sliding window to calc contrast per pixel based on neigbourhood pixel intensities, 5x5 or 7x7 is good standard to obtain resenbly good K estimation
- arg 7: CamExposureTime = 4e-3 --> cam exposure time in [sec]
- arg 8: WavelengthUm = 0.633 --> wavelength of the illumination light in [um]
- arg 9: NA = 0.1 --> numerical aperture of the objective/lens
- arg 10: Magnification = 4 --> magnification of the optical system

Explanation of the output - output consists of 4 files with specific ending attached to the original file name to indicate the type of output:
- 'sLSC.dat': summary of the used input parameters and statistics
- 'sLSC-k.tiff': contrast K map as an image stack (width and height are a bit smaller than the original), K values are represented as pixel intensity values; K is normalized so that max pixel value (16 bit depth) = max K value in the stack (it means if max K in the stack was K = 1, then in the image file its value as i pxel value is = 2^16 - 1 = 65535)
- 'sLSC-tc.tiff': (de)-correlation time tc map as an image stack (calculated from contrast, the same normalization is applied as for K); tc = TK^2 (T = cam exosure time, tc = (de-)correlation time, K = contrast), valid for T > 2tc
- 'sLSC-v.tiff': velocity v map as an image stack (calculated from tc map, the same normalization is applied as for K)

###### Example 2: Usage of Spatio-Temporal LASCA:

To process LSCI raw data stack by stLASCA type and run in the command prompt of Matlab the following function (choose a file to process by typing the respective number when asked, then press Enter):

```matlab
>> lsci_stLASCA('', 1, 100, 'ConvFilter', [200, 200, 1], 3, 10, 4e-3, 0.633, 0.28, 5)
```

Explanation of the arguments of the function:
- arg 1: InputFile = '' --> empty will cause the function to ask for a file to process
- arg 2: StartFrame = 1 --> start frame of the processing (in this case first frame, but can start from a different frame)
- arg 3: EndFrame = 100 --> end frame to process (processes all frames between StartFrame and EndFrame)
- arg 4: NumericalMethod = 'ConvFilter' --> how to calc the contrast K, available methods from slowest to fastest: 'Direct' (slowest) | 'Sums' | 'SumsVec' | 'ConvFilter' (fastest)
- arg 5: PixelXYZ = [200, 200, 1] --> [X, Y, Z] array to specify coordinate of the pixel/point where we show statistics for K (contrast) and V (velocity) in the command line report
- arg 6: XYWindowSizePx = 3 --> means 3x3 pixel window, i.e. pixel size of the XY sliding window to calc contrast per pixel based on neigbourhood pixel intensities
- arg 7: ZWindowSizePx = 10 --> number of frames - pixel size of the Z (temporal) sliding window to calc contrast per pixel along temproal (Z) direction
- arg 8: CamExposureTime = 4e-3 --> cam exposure time in [sec]
- arg 9: WavelengthUm = 0.633 --> wavelength of the illumination light in [um]
- arg 10: NA = 0.1 --> numerical aperture of the objective/lens
- arg 11: Magnification = 4 --> magnification of the optical system

Explanation of the output - output is 4 files with specific ending attached to the original file name to indicate the type of output:
- 'stLSC.dat': summary of the used input parameters and statistics
- 'stLSC-k.tiff': contrast K map as an image stack (width and height are a bit smaller than the original), K values are represented as pixel intensity values; K is normalized so that max pixel value (16 bit depth) = max K value in the stack (it means if max K in the stack was K = 1, then in the image file its value as i pxel value is = 2^16 - 1 = 65535)
- 'stLSC-tc.tiff': (de)-correlation time tc map as an image stack (calculated from contrast, the same normalization is applied as for K); tc = TK^2 (T = cam exosure time, tc = (de-)correlation time, K = contrast), valid for T > 2tc
- 'stLSC-v.tiff': velocity v map as an image stack (calculated from tc map, the same normalization is applied as for K)

Note: stLASCA uses the temproal dimension to increase the number of points used in the calculation of K, this in turn can decrease uncertainty on the cost of decreasing the temproal resolution. Also the XY sliding window can be set at smaller size, which in turn increases the spatial resolution of the output K, tc, v maps.

###### Example 3: Usage of Temporal LASCA:

To process LSCI raw data stack by tLASCA type and run in the command prompt of Matlab the following function (choose a file to process by typing the respective number when asked, then press Enter):

```matlab
>> lsci_tLASCA('', 1, 100, 'SumsVec-Discrete', [200, 200, 1], 25, 4e-3, 0.633, 0.1, 4) 
```

Explanation of the arguments of the function:
- arg 1: InputFile = '' --> empty will cause the function to ask for a file to process
- arg 2: StartFrame = 1 --> start frame of the processing (in this case first frame, but can start from a different frame)
- arg 3: EndFrame = 100 --> end frame to process (processes all frames between StartFrame and EndFrame)
- arg 4: NumericalMethod = 'SumsVec-Discrete' --> calc the contrast K, available methods: 'SumsVec-CDF' (Cumulative CDF Contrast) | 'SumsVec-Continious' (continious window sweeping) | 'SumsVec-Discrete' (discrete steps window sweeping)
- arg 5: PixelXYZ = [200, 200, 1] --> [X, Y, Z] array to specify coordinate of the pixel/point where we show statistics for K (contrast) and V (velocity) in the command line report
- arg 6: ZWindowSizePx = 25 --> number of frames - pixel size of the Z (temporal) sliding window to calc contrast per pixel along temproal (Z) direction
- arg 7: CamExposureTime = 4e-3 --> cam exposure time in [sec]
- arg 8: WavelengthUm = 0.633 --> wavelength of the illumination light in [um]
- arg 9: NA = 0.1 --> numerical aperture of the objective/lens
- arg 10: Magnification = 4 --> magnification of the optical system

Explanation of the output - output is 4 files with specific ending attached to the original file name to indicate the type of output:
- 'tLSC.dat': summary of the used input parameters and statistics
- 'tLSC-k.tiff': contrast K map as an image stack (width and height are a bit smaller than the original), K values are represented as pixel intensity values; K is normalized so that max pixel value (16 bit depth) = max K value in the stack (it means if max K in the stack was K = 1, then in the image file its value as i pxel value is = 2^16 - 1 = 65535)
- 'tLSC-tc.tiff': (de)-correlation time tc map as an image stack (calculated from contrast, the same normalization is applied as for K); tc = TK^2 (T = cam exosure time, tc = (de-)correlation time, K = contrast), valid for T > 2tc
- 'tLSC-v.tiff': velocity v map as an image stack (calculated from tc map, the same normalization is applied as for K)

###### Example 4: Usage of Multi-exposure LASCA:

To process LSCI raw data stack by meLASCA type and run in the command prompt of Matlab the following function (choose a file to process by typing the respective number when asked, then press Enter):

```matlab
>> lsci_meLASCA('', 1, 100, 4e-3, 'lin', 'on', 1, [200, 200], [20, 20], 0.633, 0.1, 'fit-lsp-contrast') 
```

Explanation of the arguments of the function:
- arg 1: InputFile = '' --> empty will cause the function to ask for a file to process
- arg 2: StartFrame = 1 --> start frame of the processing (in this case first frame, but can start from a different frame)
- arg 3: EndFrame = 100 --> end frame to process (processes all frames between StartFrame and EndFrame)
- arg 4: CamExposureTime = 4e-3 --> cam exposure time in [sec]
- arg 5: MultiExposureAlgorithm = 'lin' --> the increase of exposure time with frame number - 'off' (new dt = exposure time) | 'lin' (new dt = dt + exposure time) | 'pow2' (new dt = 2dt)
- arg 6: MultiExposureGroupFrames = 'on' --> 'on' (group frames per multi-exposure time to increase accuracy) | 'off' (no grouping of frames per multi-exposure time; much less accurate)
- arg 7: FrameRate = 1 --> desired new output fps (frames per second) of the multi-exposure stack, e.g. 1 fps (must be at least 2x less than the original, the best is 10-100 bigger). This means if the input fps = 250, T (exposure time) = 4 ms, then multi-exposure frames will be calc up to new T = 1 sec.
- arg 8: PixelXY = [200, 200] --> [X, Y] or [cols, rows] (coordinate of the point where we calc K(T), Ct(tau), Velocity etc)
- arg 9: XYWindowSizePx = [20, 20] --> [X, Y] or [cols, rows] size in pixels of the window over which we calc average contrast K
- arg 10: WavelengthUm = 0.633 --> wavelength of the illumination light in [um]
- arg 11: NA = 0.1 --> numerical aperture of the objective/lens
- arg 12: FitMethodForVelocity = 'fit-lsp-contrast' --> fit method to extract velocity --> 'fit-autocovariance' (fit Autocovariance vs Tau curve) | 'fit-lsp-contrast' (fit LSP Contrast vs Exposure Time curve)

Explanation of the output - output is 4 files with specific ending attached to the original file name to indicate the type of output:
- 'me=lin(gf=on)_Ct=(200,200)(20x20)_wl=0.633um_NA=0.10_fps=1.0.dat': contains the calculated Ct vs tau curve (as csv tables)
- 'me=lin(gf=on)_K=(200,200)(20x20)_wl=0.633um_NA=0.10_fps=1.0.dat': contains the calculated K vs Time (exposure time) curve and Kfit vs Time (as csv tables)
- 'me=lin(gf=on)_PSD=(200,200)(20x20)_wl=0.633um_NA=0.10_fps=1.0.dat': contains the calculated PSD (Power Spectral Density) vs frequency curve (as csv tables)
- 'me=lin(gf=on)_V=(200,200)(20x20)_wl=0.633um_NA=0.10_fps=1.0': contains summary of the fitted parameters (v, tc, beta, ro, K noise) and their lower and upper confidence intervals; this file is the main result of the processing with meLASCA

Note: for meLASCA to work correctly the min exposure time must be equal to 1/fps - this is since from a stack with fixed exposure time we generate a multi exposure stack. Essentially this implements so called sMESI (Single Multi-exposure Speckle Imaging). This meLASCA gives the most accurate and relaible results and takes into account noise as well. However, it is very slow and time consuming, so it is usually used for getting K, tc, v in single spatial points or handful of points. The reason is that it makes fitting, which for large image slow downs dramatically to overall performance.

###### Example 5: Usage of (Temporal) Frequency Domain Laser Speckle Imaging analysis:

To process LSCI raw data stack by tFDLSI type and run in the command prompt of Matlab the following function (choose a file to process by typing the respective number when asked, then press Enter):

```matlab
>> lsci_tFDLSI('', 'xcov', 1, 100, [200, 200, 1], 100, 4e-3, 250, 0.633, 0.1, 4) 
```

Explanation of the arguments of the function:
- arg 1: InputFile = '' --> empty will cause the function to ask for a file to process
- arg 2: NumericalMethod = 'xcov' --> calc Ct(tau) using 'fft' (Fast Fourier Transform) or 'xcov' (built in matlab (auto)-covariance function)
- arg 3: StartFrame = 1 --> start frame of the processing (in this case first frame, but can start from a different frame)
- arg 4: EndFrame = 100 --> end frame to process (processes all frames between StartFrame and EndFrame)
- arg 5: PixelXYZ = [200, 200, 1] --> [X, Y, Z] array to specify coordinate of the pixel/point where we show statistics for K (contrast) and V (velocity) in the command line report
- arg 6: ZWindowSizeFrames = 100 --> pixel size of the Z sliding window to calc the autocovariance Ct(tau)
- arg 7: CamExposureTime = 4e-3 --> cam exposure time in [sec]
- arg 8: FrameRate = 250 --> frames per second (sampling frequency)
- arg 9: WavelengthUm = 0.633 --> wavelength of the illumination light in [um]
- arg 10: NA = 0.1 --> numerical aperture of the objective/lens
- arg 11: Magnification = 4 --> magnification of the optical system

Explanation of the output - output is 4 files with specific ending attached to the original file name to indicate the type of output:
- 'tFDLS-tc.tiff': the correlation time tc map as (multi-page) tiff file
- 'tFDLS-vs.tiff': the velocity map (vs - assumes single flow velocity with no diffusion model) as (multi-page) tiff file
- 'tFDLS-v0.tiff': the velocity map (v0 - directed flow velocity; the model assumes difussing tracers) as (multi-page) tiff file
- 'tFDLS-vd.tiff': the velocity map (vd - flow velocity due to diffusion; the model assumes difussing tracers) as (multi-page) tiff file

###### Example 6: Usage of (Temporal) Entropy LASEA:

To process LSCI raw data stack by teLASEA type and run in the command prompt of Matlab the following function (choose a file to process by typing the respective number when asked, then press Enter):

```matlab
>> lsci_teLASEA('', 1, 100, 'Balanced', [200, 200, 1], 100, 8, 4e-3) 
```

Explanation of the arguments of the function:
- arg 1: InputFile = '' --> empty will cause the function to ask for a file to process
- arg 2: StartFrame = 1 --> start frame of the processing (in this case first frame, but can start from a different frame)
- arg 3: EndFrame = 100 --> end frame to process (processes all frames between StartFrame and EndFrame)
- arg 4: EntropyEstimator = 'Balanced' --> 'Miller' | 'Balanced' (Miller = improved naive entropy estimator for many frames; Balanced = balanced entropy estimator for low frame number, e.g. below 100)
- arg 5: PixelXYZ = [200, 200, 1] --> [X, Y, Z] array to specify coordinate of the pixel/point where we show statistics for H (entropy) and V (velocity) in the command line report
- arg 6: ZWindowSizePx = 100 --> 25, 50, 100 etc (pixel size of the Z sliding window to calc entropy per pixel)
- arg 7: BitDepth = 8 --> entropy bit dept with respect to max pix value of the input image: 8 | 16 bits data values
- arg 8: CamExposureTime = 4e-3 --> cam exposure time in [sec]

Explanation of the output - output is 4 files with specific ending attached to the original file name to indicate the type of output:
- 'teLSE.dat':  summary of the used input parameters and statistics
- 'teLSE-h.tiff': the entropy h map as tiff file (normalization applied to max pix value)
- 'teLSE-v.tiff': the velocity v map as tiff file (normalization applied), to calc real velocity use the info from the summary

*Note: teLASEA method is still not clear how good it can calc the velocity, use for testing purposes only to play with different estimator than contrast K.

###### Example 7: Usage of Fourier Transform Laser Speckle Analysis:

To process LSCI raw data stack by lsci_fftLSA type and run in the command prompt of Matlab the following function (choose a file to process by typing the respective number when asked, then press Enter):

```matlab
>> lsci_fftLSA('', 'xcov', 1, 100, 100, 4e-3, 250, [200, 200], [20, 20], 6.4, 4, 0.633, 0.1) 
```

Explanation of the arguments of the function:
- arg 1: InputFile = '' --> empty will cause the function to ask for a file to process
- arg 2: NumericalMethod = 'xcov' --> calc Ct(tau) using 'fft' (Fast Fourier Transform) or 'xcov' (built in matlab (auto)-covariance function)
- arg 3: StartFrame = 1 --> start frame of the processing (in this case first frame, but can start from a different frame)
- arg 4: EndFrame = 100 --> end frame to process (processes all frames between StartFrame and EndFrame)
- arg 5: ZWindowSizeFrames = 100 --> pixel size of the Z sliding window to calc the autocovariance Ct(tau)
- arg 6: CamExposureTime = 4e-3 --> cam exposure time in [sec]
- arg 7: FrameRate = 250 --> frames per second of the acquired stack
- arg 8: PixelXY = [200, 200] --> [X, Y] coordinate of the point where we calc FFT, Ct(tau) etc
- arg 9: PixelWindowXY = [20, 20] --> [X, Y] size in pixels over which we average the Ct(tau)
- arg 10: CamPixelSizeUm = 6.4 --> physical sze of a pixel in [um]
- arg 11: Magnification = 4 --> magnification of the optical system
- arg 12: WavelengthUm = 0.633 --> wavelength of the illumination light in [um]
- arg 13: NA = 0.1 --> numerical aperture of the objective/lens

Explanation of the output - output is 4 files with specific ending attached to the original file name to indicate the type of output:
- few graphs: autocovariance + fit to get velocities, cross-covariance, PSD, signal intensity along z etc
- 'fftLSA_Ct=(200,200)(20x20).dat':  contains csv of the Ct ex (experiment/raw data) vs tau, CtV0 vs tau (fit assuming v0 is single velocity, no diffusion), CtV0Vd vs tau (fit assuming directed flow + diffusion model)
- 'fftLSA_PSD=(200,200)(20x20).dat': contains PSD vs freq calculated along z (temporal direction) using the intensity values of the raw data (averaging applied for many such curves defined by the XY window size)
- 'fftLSA_V=(200,200)(20x20).dat': contains summary of the extracted from the different fits velocities vs, v0, vd + their lower and upper bound (95% confidence interval)

## References and Literature

1. Stoyan Yordanov et al, "Real-time monitoring of biomechanical activity in aphids by laser speckle contrast imaging", DOI: https://doi.org/10.1364/OE.431989)
2. W. James Tom et al, "Efficient Processing of Laser Speckle Contrast Images", DOI: https://doi.org/10.1109/TMI.2008.925081
3. Julio C. Ramirez-San-Juan et al, "Impact of velocity distribution assumption on simplified laser speckle imaging equation ", DOI: https://doi.org/10.1364/OE.16.003197
4. P. Miao et al, "Random process estimator for laser speckle imaging of cerebral blood flow", DOI: https://doi.org/10.1364/OE.18.000218
5. Oliver Thomson et al, "Tissue perfusion measurements: multiple-exposure laser speckle analysis generates laser Doppler-like spectra", DOI: https://doi.org/10.1117/1.3400721
6. Hao Li et al, "Directly measuring absolute flow speed by frequency-domain laser speckle imaging", DOI: https://doi.org/10.1364/OE.22.021079
7. Tomi Smausz et al, "Real correlation time measurement in laser speckle contrast analysis using wide exposure time range images", DOI: https://doi.org/10.1364/AO.48.001425
8. Ashwin B. Parthasarathy et al, "Robust flow measurement with multi-exposure speckle imaging", DOI: https://doi.org/10.1364/OE.16.001975
9. Ping Kong et al, "A novel highly efficient algorithm for laser speckle imaging", DOI: https://doi.org/10.1016/j.ijleo.2016.04.004
10. S. Kazmi et al, "Evaluating multi-exposure speckle imaging estimates of absolute autocorrelation times", DOI: https://doi.org/10.1364/OL.40.003643
11. D. Boas et al, "Laser speckle contrast imaging in biomedical optics", DOI: https://doi.org/10.1117/1.3285504
12. Mitchell A. Davis et al, "Sensitivity of laser speckle contrast imaging to flow perturbations in the cortex", DOI: https://doi.org/10.1364/BOE.7.000759
13. Ashwin B. Parthasarathy et al, "Quantitative imaging of ischemic stroke through thinned skull in mice with Multi Exposure Speckle Imaging", DOI: https://doi.org/10.1364/BOE.1.000246
14. Annemarie Nadort et al, "Quantitative laser speckle flowmetry of the in vivo microcirculation using sidestream dark field microscopy", DOI: https://doi.org/10.1364/BOE.4.002347
15. Peng Miao et al, "Entropy analysis reveals a simple linear relation", DOI: http://dx.doi.org/10.1364/OL.39.003907
16. Juan A Bonachela et al, "Entropy estimates of small data sets", DOI: http://dx.doi.org/10.1088/1751-8113/41/20/202001
