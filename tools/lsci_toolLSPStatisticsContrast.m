function lsci_toolLSPStatisticsContrast(InputFile, StartFrame, EndFrame, HistogramBins, PixelXYZ, XYWindowSizePx, ZWindowSizePx)
% Plots and checks the type of PDF (Probability Density Function) of Contrast in an image pixel or area follows (spatial or temproal domain)
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

% Check for given PDF type
lspContrastXYZ = stLASCASumsVectorized(inXYZFrames, XYWindowSizePx, ZWindowSizePx); % calc LSP Contrast
subFramesXYZ = getSubFrameXYZ(lspContrastXYZ, XYWindowSizePx, ZWindowSizePx, pixX, pixY, pixZ); % get subframe of the input data
PdfXY = pdfLSPContrastLognormal(subFramesXYZ, HistogramBins);    

% Show histogram
showHistogram(PdfXY, subFramesXYZ, HistogramBins);

% Set base file names
baseFileNameLSPContrast = fullfile(filePath, [fileName '_stLSC-k']); % Assemble tiff file name for Laser Speckle Contrast

% Save result for Laser Speckle Contrast as tiff
type3DStackItNormalization = 'global';
outputFileType = 'tiff';
lsci_SaveToFrames(lspContrastXYZ, baseFileNameLSPContrast, outputFileType, type3DStackItNormalization);
 
% Show progress and stats
elapsedTime1 = toc(startTime1);

fprintf('\nEnd of processing --> Start Frame = %d, End Frame = %d\n', StartFrame, EndFrame); % show progress
fprintf('Statistics Contrast --> Kmax = %.3f, Kmin = %.3f, Kmean = %.3f, Kstd = %.3f\n', max(subFramesXYZ, [], 'all'), min(subFramesXYZ, [], 'all'), mean(subFramesXYZ, 'all'), std(subFramesXYZ, 0, 'all')); % show progress
fprintf('Processing time = %.3f [sec]\n\n', elapsedTime1);

end

function rtrnPdf = pdfLSPContrastLognormal(LSPContrastXYZ, HistogramBins)
% Return Lognormal PDF distribution of the contrast values
% See the following paper for more details:
% Kosar Khaksari eet al, "Laser Speckle Modeling and Simulation for Biophysical Dynamics: Influence of Sample Statistics", DOI: http://doi.org/10.18287/JBPE17.03.040302

rtrnPdf = struct();
rtrnPdf.X = []; % vector of X values (binned values)
rtrnPdf.Y = []; % vector of Y values (counts/freq. occurence)

% Set X
Kmin = min(LSPContrastXYZ, [], 'all');
Kmax = max(LSPContrastXYZ, [], 'all');
Kmean = mean(LSPContrastXYZ, 'all');
Kvar = var(LSPContrastXYZ, 0, 'all');
rtrnPdf.X = Kmin:((Kmax-Kmin)/HistogramBins):Kmax;

% Calc distribution parameters mu and sigma
mu = log(Kmean^2/sqrt(Kvar + Kmean^2));
sigma = sqrt(log(1 + Kvar/Kmean^2));

% Fit to lognormal distribution in order to obtain mu and sigma of the distribution
% pdLognormal = fitdist(LSPContrastXYZ(:), 'Lognormal');
% mu = pdLognormal.mu;
% sigma = pdLognormal.sigma;

rtrnPdf.Y = (1./(rtrnPdf.X*sigma*sqrt(2*pi))) .* exp(- (log(rtrnPdf.X) - mu).^2 ./ (2*sigma^2));

end

function rtrnXYZLSPContrast = stLASCASumsVectorized(InXYZFrames, XYWindowSizePx, ZWindowSizeFrames)
% Sums method (vectorized along Z) to calc the Laser Speckle Contrast

[lengthX, lengthY, lengthZ] = size(InXYZFrames);

% Pre-allocate
rtrnXYZLSPContrast = zeros((lengthX - XYWindowSizePx), (lengthY - XYWindowSizePx), (lengthZ - ZWindowSizeFrames));
meanIntensityXYZ = zeros((lengthX - XYWindowSizePx), (lengthY - XYWindowSizePx), lengthZ);
meanSqrIntensityXYZ = zeros((lengthX - XYWindowSizePx), (lengthY - XYWindowSizePx), lengthZ);

% Calc Laser Speckle Contrast map --> k = std(I)/<I> = sqrt(<I^2> - <I>^2)/<I> = sqrt(<I^2>/<I>^2 - 1)
fprintf('\nProgress LSP Contrast (SumsVec Calc) (Step 1 of 2): 000.0 [%%] | 00000.0 [sec]');
for iZ = 1:(lengthZ - ZWindowSizeFrames) % loop throughout frames
    startTime = tic;
    
    % Precalc mean intensity and mean squared intensity map for each frame (calc only ones)
    if (iZ == 1)
        for iX = 1:(lengthX - XYWindowSizePx) % loop through image height            
            for iY = 1:(lengthY - XYWindowSizePx) % loop through image width
                subFrames = InXYZFrames(iX:(iX + XYWindowSizePx - 1), iY:(iY + XYWindowSizePx - 1), :); % extract subframes XxYxZ given by the window size
                meanIntensityXYZ(iX, iY, 1:end) = sum(subFrames, [1, 2])./(XYWindowSizePx^2); % mean XY intensity along Z
                meanSqrIntensityXYZ(iX, iY, 1:end) = sum(subFrames .^2, [1, 2])./(XYWindowSizePx^2); % mean of squared XY intensities along Z
                %meanSqrIntensity(iX, iY, 1:end) = sum(subFrames .^2, [1, 2])./(XYWindowSizePx^2 - 1); % mean of squared XY intensities along Z               
            end
            fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'); % delete previous progress line
            fprintf('(Step 1 of 2): %05.1f [%%] | %07.1f [sec]', (iX/(lengthX - XYWindowSizePx))*100, ((lengthX - XYWindowSizePx) - iX)*(toc(startTime)/iX));
        end
    end
    
    % Average the mean XY intensities over the temporal window Z
    subFramesMeanIntensity = meanIntensityXYZ(:, :, iZ:(iZ + ZWindowSizeFrames - 1)); % extract XY frame with Z subframe
    meanIntensityZ = sum(subFramesMeanIntensity, 3)./ZWindowSizeFrames; % mean of the mean XY intensity along Z window
    subFramesMeanSqrIntensity = meanSqrIntensityXYZ(:, :, iZ:(iZ + ZWindowSizeFrames - 1)); % extract XY frame with Z subframe
    meanSqrIntensityZ = sum(subFramesMeanSqrIntensity, 3)./ZWindowSizeFrames; % mean of the mean of squared XY intensities along Z
    rtrnXYZLSPContrast(1:end, 1:end, iZ) = sqrt(meanSqrIntensityZ - meanIntensityZ .^2)./meanIntensityZ; % calc contrast
    
    % Calc Elapsed time and show progress
    elapsedTime = toc(startTime);
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'); % delete previous progress line
    fprintf('(Step 2 of 2): %05.1f [%%] | %07.1f [sec]', (iZ/(lengthZ - ZWindowSizeFrames))*100, ((lengthZ - ZWindowSizeFrames) - iZ)*elapsedTime);
end

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
plot(gca, PdfXY.X, PdfXY.Y, '-.', 'LineWidth', 2, 'DisplayName', 'PDF Theory (Lognormal)', 'Color', [0 0.5 1]);

set(gca, 'FontSize', 10); % set font size
set(gca, 'XScale', 'linear'); % set scale
set(gca, 'Box', 'on'); % set plot to a box
title(gca, 'Histogram of Contrast Distribution');
xlabel(gca, 'Contrast [Bins]');
ylabel(gca, 'Pdf');

legend;

hold off;

end
