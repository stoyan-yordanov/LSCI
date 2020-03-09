function lsci_toolLSPStatisticsIntensity(InputFile, StartFrame, EndFrame, HistogramBins, PixelXYZ, XYWindowSizePx, ZWindowSizePx)
% Plots and checks the type of PDF (Probability Density Function) of Intensity in an image pixel or area follows (spatial or temproal domain)
% InputFile = file name of the input data (if empty brings command line file dialog) - supports avi (video) | mj2 (video Motion Jpeg 2000) | tiff (multipage)
% Process between StartFrame and EndFrame frames.
% HistogramBins = number of histogram bins
% PixelXYZ = [X, Y, Z] (coordinate of the point where we show statistics
% XYWindowSizePx = XY size of the window in px where we probe the intensity
% ZWindowSizePx = Z size of the window in px where we probe the intensity

fprintf('\nStart Checking Laser Speckle PDF... \n'); % show progress
startTime1 = tic;

% Single XY pixel location
pixY = PixelXYZ(1);
pixX = PixelXYZ(2);
pixZ = PixelXYZ(3);

% Case no file provided or the string is not a valid file --> Get dir file list and choose a file t process
if strcmp(InputFile, '') || ~isfile(InputFile)    
    fileDirFilter = '*';
    fileList = lsci_sysGetDirectoryFileList(fileDirFilter); % return the list of file in the current dir
    fileList = lsci_sysChooseFilesFromFileList(fileList); % get the file(s) to be processed
    InputFile = fileList{1, 1}; % only one file (the first one) will be processed
end

% Check file name
[filePath, fileName, fileExtension] = fileparts(InputFile);

% Read input raw (video/image) frames
inXYZFrames = lsci_ReaderFramesToMatrix(InputFile, StartFrame, EndFrame, 'double'); % XY images array (Z = frame index)

% Calc statistics on a subframe for given PDF type
subFramesXYZ = getSubFrameXYZ(inXYZFrames, XYWindowSizePx, ZWindowSizePx, pixX, pixY, pixZ); % get subframe of the input data
PdfXY = pdfLSPIntensityRayleigh(subFramesXYZ, HistogramBins);

% Show histogram
showHistogram(PdfXY, subFramesXYZ, HistogramBins);
 
% Show progress and stats
elapsedTime1 = toc(startTime1);

fprintf('\nEnd of processing --> Start Frame = %d, End Frame = %d\n', StartFrame, EndFrame); % show progress
fprintf('Statistics Intensity --> Imax = %.1f, Imean = %.1f, Istd = %.1f\n', max(subFramesXYZ, [], 'all'), mean(subFramesXYZ, 'all'), std(subFramesXYZ, 0, 'all')); % show progress
fprintf('Processing time = %.3f [sec]\n\n', elapsedTime1);

end

function rtrnPdf = pdfLSPIntensityRayleigh(LSPIntensityXYZ, HistogramBins)
% Return Rayleight PDF distribution of the intensity
% See the following paper for more details:
% Kosar Khaksari eet al, "Laser Speckle Modeling and Simulation for Biophysical Dynamics: Influence of Sample Statistics", DOI: http://doi.org/10.18287/JBPE17.03.040302

rtrnPdf = struct();
rtrnPdf.X = []; % vector of X values (binned intenisty values)
rtrnPdf.Y = []; % vector of Y values (intensity counts/freq. occurence)

% Set X
Imin = min(LSPIntensityXYZ, [], 'all');
Imax = max(LSPIntensityXYZ, [], 'all');
rtrnPdf.X = Imin:((Imax-Imin)/HistogramBins):Imax;

% Get Rayleigh distribution given the scale parameter --> pdf(I) = (1/<I>) * exp(- I/<I>)
Imean = mean(LSPIntensityXYZ, 'all');
rtrnPdf.Y = (1/Imean) .* exp(- rtrnPdf.X ./ Imean);

end

function rtrnSubFrameXYZ = getSubFrameXYZ(InXYZFrames, XYWindowSizePx, ZWindowSizePx, pixX, pixY, pixZ)
% Return a subframe of the input data

[rows, cols, frames] = size(InXYZFrames);

% Get sub frame of the input data
pxX.min = pixX - ceil(XYWindowSizePx/2) + 1;
pxX.max = pixX + floor(XYWindowSizePx/2);

pxY.min = pixY - ceil(XYWindowSizePx/2) + 1;
pxY.max = pixY + floor(XYWindowSizePx/2);

pxZ.min = pixZ;
pxZ.max = pixZ + ZWindowSizePx - 1;

if (pxX.min > 0 && pxX.max <= rows) && (pxY.min > 0 && pxY.max <= cols) && (pxZ.min > 0 && pxZ.max <= frames)    
    rtrnSubFrameXYZ = InXYZFrames(pxX.min:pxX.max, pxY.min:pxY.max, pxZ.min:pxZ.max);
else
    fprintf('\nSome of the XYZ Frams indexes is out of range -- > Check index range is within allowed interval\n');
    error('Exit due to the above error!');
end

end

function showHistogram(PdfXY, FrameXYZ, HistogramBins)
% Show histogram of the input data and original theory PDF

[rows, cols, frames] = size(FrameXYZ);

figure;
hold on;
histogram(gca, FrameXYZ, HistogramBins, 'Normalization', 'pdf', 'DisplayName', sprintf('PDF Experiment (xyz=%dx%dx%d)', cols, rows, frames), 'FaceColor', [0 0.5 0.5]);
plot(gca, PdfXY.X, PdfXY.Y, '-.', 'LineWidth', 2, 'DisplayName', 'PDF Theory (Rayleigh)', 'Color', [0 0.5 1]);

set(gca, 'FontSize', 10); % set font size
set(gca, 'XScale', 'linear'); % set scale
set(gca, 'Box', 'on'); % set plot to a box
title(gca, 'Histogram of Intensity Distribution');
xlabel(gca, 'Intensity [Bins]');
ylabel(gca, 'Pdf');

legend;

hold off;

end
