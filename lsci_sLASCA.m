function lsci_sLASCA(InputFile, StartFrame, EndFrame, NumericalMethod, PixelXYZ, XYWindowSizePx, CamExposureTime, WavelengthUm, NA, Magnification)
% Read video, calc Spatial Laser Speckle Contrast and save it as multi page tiff file.
% InputFile = file name of the input data (if empty brings command line file dialog) - supports avi (video) | mj2 (video Motion Jpeg 2000) | tiff (multipage)
% Process between StartFrame and EndFrame frames.
% NumericalMethod = 'Direct' | 'Sums' | 'SumsVec' | 'ConvFilter' (fastest)
% PixelXYZ = [X, Y, Z] (coordinate of the point where we show statistics (K and V)
% XYWindowSizePx = 5 (i.e. 5x5 px), 7 (i.e. 7x7 px) etc (pixel size of the XY sliding window to calc LSC per pixel based on neigbourhood pixel intensities)
% CamExposureTime = 250e-6 [sec] etc (cam exposure time in sec.)
% WavelengthUm = wavelength of the illumination light in [um]
% NA = numerical aperture
% Magnification = magnification of the optical system


% Single XY pixel location to calc/show/save K and V
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

startTime = tic;

% Check file name
[filePath, fileName, fileExtension] = fileparts(InputFile);

% Read input raw frames
inXYZFrames = lsci_ReaderFramesToMatrix(InputFile, StartFrame, EndFrame, 'double'); % XY images array (Z = frame index)

% The structure that will store the results of the processing
dataLSP = struct();
dataLSP.Contrast = []; % LSP Contrast --> represents K
dataLSP.CorrelationTime = []; % Correlation Time --> represents tc
dataLSP.Velocity = []; % Velocity --> represents V

% Write frames to multipage tiff file
fprintf('\nStart calculating Spatial Laser Speckle Contrast... \n'); % show progress

% Process input and calc Laser Speckle Contrast Map
dataLSP.Contrast = lsciProcessLASCAInputByMethod(inXYZFrames, NumericalMethod, XYWindowSizePx);

% Process Laser Speckle Contrast and calc Tc map in [s]
dataLSP.CorrelationTime = calcCorrelationTimeMap(dataLSP.Contrast, CamExposureTime);

% Process Laser Speckle Tc and calc Velocity map in [um/s]
dataLSP.Velocity = calcVelocityMap(dataLSP.CorrelationTime, WavelengthUm, NA, Magnification);

% Set base file names
baseFileNameLSPContrast = fullfile(filePath, [fileName '_sLSC-k']); % Assemble base file name for Laser Speckle Contrast
baseFileNameLSPCorrelationTime = fullfile(filePath, [fileName '_sLSC-tc']); % Assemble tiff file name for Laser Speckle Correlation Time
baseFileNameLSPVelocity = fullfile(filePath, [fileName '_sLSC-v']); % Assemble base file name for Laser Speckle Velocity

% Save result for Laser Speckle Contrast as tiff
type3DStackItNormalization = 'global';
outputFileType = 'tiff';
lsci_SaveToFrames(dataLSP.Contrast, baseFileNameLSPContrast, outputFileType, type3DStackItNormalization);
lsci_SaveToFrames(dataLSP.CorrelationTime, baseFileNameLSPCorrelationTime, outputFileType, type3DStackItNormalization);
lsci_SaveToFrames(dataLSP.Velocity, baseFileNameLSPVelocity, outputFileType, type3DStackItNormalization);

% Save processed data
saveLspData(dataLSP, InputFile, '_sLSC', PixelXYZ, XYWindowSizePx, CamExposureTime, NumericalMethod, WavelengthUm, NA, Magnification);

% Show elapsed time and progress
elapsedTime = toc(startTime);

fprintf('\n\nEnd of processing --> Start Frame = %d, End Frame = %d\n', StartFrame, EndFrame); % show progress
fprintf('Statistics --> Kmax = %f, Kmin = %f, Kmean = %f\n', max(dataLSP.Contrast, [], 'all'), min(dataLSP.Contrast, [], 'all'), mean(dataLSP.Contrast, 'all')); % show progress
fprintf('Statistics --> Tcmax = %.3g [s], Tcmin = %.3g [s], Tcmean = %.3g [s]\n', max(dataLSP.CorrelationTime, [], 'all'), min(dataLSP.CorrelationTime, [], 'all'), mean(dataLSP.CorrelationTime, 'all')); % show progress
fprintf('Statistics --> Vmax = %.3f [mm/s], Vmin = %.3f [mm/s], Vmean = %.3f [mm/s]\n', max(dataLSP.Velocity, [], 'all')/1000, min(dataLSP.Velocity, [], 'all')/1000, mean(dataLSP.Velocity, 'all')/1000); % show progress
fprintf('Statistics Pixel[%d, %d, %d] --> K = %f, Tc = %g [s], V = %.3f [mm/s]\n', pixY, pixX, pixZ, dataLSP.Contrast(pixX, pixY, pixZ), dataLSP.CorrelationTime(pixX, pixY, pixZ), dataLSP.Velocity(pixX, pixY, pixZ)/1000); % show progress
fprintf('Processing time = %f [sec]\n\n', elapsedTime);

% For debugging
% fprintf('\nVmax:\n');
% Vmax1 = max(dataLSP.Velocity, [], 'all');
% Vmax2 = max(dataLSP.Velocity(:)./Vmax1, [], 'all'); % normalize to 1
% cInt = 2^16 - 1; % coefficient to convert to 8 or 16 bit integer
% Vmax3 = (cInt.*Vmax2); % convert to 8 or 16 bit depth
% fprintf('Vmax1 = %f, Vmax2 = %f, Vmax3 = %f\n', Vmax1, Vmax2, Vmax3);
% 
% fprintf('\nVmin:\n');
% Vmin1 = min(dataLSP.Velocity, [], 'all');
% Vmin2 = min(dataLSP.Velocity(:)./Vmax1, [], 'all'); % normalize to 1
% cInt = 2^16 - 1; % coefficient to convert to 8 or 16 bit integer
% Vmin3 = (cInt.*Vmin2); % convert to 8 or 16 bit depth
% fprintf('Vmin1 = %f, Vmin2 = %f, Vmin3 = %f\n', Vmin1, Vmin2, Vmin3);
% 
% fprintf('\nVmax + Vmin [file re-read]:');
% VinXYZFrames = lsci_ReaderFramesToMatrix([baseFileNameLSPVelocity '.' outputFileType], StartFrame, EndFrame, 'double'); % XY images array (Z = frame index)
% Vmax4 = max(VinXYZFrames, [], 'all');
% Vmin4 = min(VinXYZFrames, [], 'all');
% fprintf('Vmax4 = %f\n', Vmax4);
% fprintf('Vmin4 = %f\n', Vmin4);

end

function rtrnXYZLSPContrast = lsciProcessLASCAInputByMethod(InXYZFrames, NumericalMethod, XYWindowSizePx)
% Calc Spatial Laser Speckle Contrast and Velocity maps
% Numerical algorithms take ideas from the following papers:
% W. James Tom et al, "Efficient Processing of Laser Speckle Contrast Images", DOI link: https://doi.org/10.1109/TMI.2008.925081

% Calc Laser Speckle Contrast map
switch(NumericalMethod)
    case 'Direct' % Slowest speed
        rtrnXYZLSPContrast = sLASCADirect(InXYZFrames, XYWindowSizePx);
    case 'Sums'
        rtrnXYZLSPContrast = sLASCASums(InXYZFrames, XYWindowSizePx);
    case 'SumsVec'
        rtrnXYZLSPContrast = sLASCASumsVectorized(InXYZFrames, XYWindowSizePx);
    case 'ConvFilter'
        rtrnXYZLSPContrast = sLASCAConvFilter(InXYZFrames, XYWindowSizePx);
    otherwise
        fprintf('\n\nUnsupported numerical method --> %s\n', NumericalMethod);
        error('Exit due to error!');
end

% Filter Contrast by removing/replacing all K > 1 and K = NaN
rtrnXYZLSPContrast(rtrnXYZLSPContrast > 1) = 1;
rtrnXYZLSPContrast(isnan(rtrnXYZLSPContrast)) = 1;

end

function rtrnXYZLSPContrast = sLASCADirect(InXYZFrames, XYWindowSizePx)
% Direct calc of Laser Speckle Contrast

[lengthX, lengthY, lengthZ] = size(InXYZFrames);

% Pre-allocate
rtrnXYZLSPContrast = zeros((lengthX - XYWindowSizePx + 1), (lengthY - XYWindowSizePx + 1), lengthZ);

% Calc Laser Speckle Contrast map --> k = std(I)/<I>
fprintf('\nProgress LSP Contrast (Direct Calc): 000.0 [%%] | 00000.0 [sec]');
%workerThreads = 4;
%parfor (iZ = 1:lengthZ, workerThreads) % parallel for-loop through frames
for iZ = 1:lengthZ % loop thorough frames
    startTime = tic;
    % Calc Laser Speckle Contrast for each frame
    for iX = 1:(lengthX - XYWindowSizePx + 1) % loop through image height
        for iY = 1:(lengthY - XYWindowSizePx + 1) % loop through image width
            subFrame = InXYZFrames(iX:(iX + XYWindowSizePx - 1), iY:(iY + XYWindowSizePx - 1), iZ); % extract subframe given by the window size
            %meanIntensity = mean(subFrame, 'all'); % mean intensity
            meanIntensity = sum(subFrame, 'all')/(XYWindowSizePx^2); % mean intensity
            stdIntensity = std(subFrame, 0, 'all'); % standard deviation
            rtrnXYZLSPContrast(iX, iY, iZ) = stdIntensity/meanIntensity; % calc contrast for the given pixel (actually the center pixel of the window)
        end
    end
    elapsedTime = toc(startTime);
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'); % delete previous progress line
    fprintf('%05.1f [%%] | %07.1f [sec]', (iZ/lengthZ)*100, (lengthZ-iZ)*elapsedTime);
end

end

function rtrnXYZLSPContrast = sLASCASums(InXYZFrames, XYWindowSizePx)
% Sums method to calc the Laser Speckle Contrast

[lengthX, lengthY, lengthZ] = size(InXYZFrames);

% Pre-allocate
rtrnXYZLSPContrast = zeros((lengthX - XYWindowSizePx + 1), (lengthY - XYWindowSizePx + 1), lengthZ);

% Calc Laser Speckle Contrast map --> k = std(I)/<I> = sqrt(<I^2> - <I>^2)/<I> = sqrt(<I^2>/<I>^2 - 1)
fprintf('\nProgress LSP Contrast (Sums Calc): 000.0 [%%] | 00000.0 [sec]');
%workerThreads = 4;
%parfor (iZ = 1:lengthZ, workerThreads) % loop thorough frames
for iZ = 1:lengthZ % loop thorough frames
    startTime = tic;
    % Calc Laser Speckle Contrast for each frame
    for iX = 1:(lengthX - XYWindowSizePx + 1) % loop through image height
        for iY = 1:(lengthY - XYWindowSizePx + 1) % loop through image width
            subFrame = InXYZFrames(iX:(iX + XYWindowSizePx - 1), iY:(iY + XYWindowSizePx - 1), iZ); % extract subframe given by the window size
            meanIntensity = sum(subFrame, 'all')/(XYWindowSizePx^2); % mean intensity
            meanSqrIntensity = sum(subFrame .^2, 'all')/(XYWindowSizePx^2); % mean of squared intensities
            %meanSqrIntensity = sum(subFrame .^2, 'all')/(XYWindowSizePx^2 - 1); % mean of squared intensities 
            rtrnXYZLSPContrast(iX, iY, iZ) = sqrt(meanSqrIntensity - meanIntensity^2)/meanIntensity; % calc contrast for the given pixel (actually the center pixel of the window)            
        end
    end
    elapsedTime = toc(startTime);
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'); % delete previous progress line
    fprintf('%05.1f [%%] | %07.1f [sec]', (iZ/lengthZ)*100, (lengthZ-iZ)*elapsedTime);
end

end

function rtrnXYZLSPContrast = sLASCASumsVectorized(InXYZFrames, XYWindowSizePx)
% Sums method (vectorized along Z) to calc the Laser Speckle Contrast

[lengthX, lengthY, lengthZ] = size(InXYZFrames);

% Pre-allocate
rtrnXYZLSPContrast = zeros((lengthX - XYWindowSizePx + 1), (lengthY - XYWindowSizePx + 1), lengthZ);

% Calc Laser Speckle Contrast map --> k = std(I)/<I> = sqrt(<I^2> - <I>^2)/<I> = sqrt(<I^2>/<I>^2 - 1)
fprintf('\nProgress LSP Contrast (Sums Vectorized Calc): 000.0 [%%] | 00000.0 [sec]');
for iX = 1:(lengthX - XYWindowSizePx + 1) % loop through image height
    startTime = tic;
    for iY = 1:(lengthY - XYWindowSizePx + 1) % loop through image width
        subFrames = InXYZFrames(iX:(iX + XYWindowSizePx - 1), iY:(iY + XYWindowSizePx - 1), :); % extract subframes XxYxZ given by the window size
        meanIntensity = sum(subFrames, [1, 2])./(XYWindowSizePx^2); % mean intensity along Z
        meanSqrIntensity = sum(subFrames .^2, [1, 2])./(XYWindowSizePx^2); % mean of squared intensities along Z
        %meanSqrIntensity = sum(subFrames .^2, [1, 2])./(XYWindowSizePx^2 - 1); % mean of squared intensities along Z
        rtrnXYZLSPContrast(iX, iY, 1:end) = sqrt(meanSqrIntensity - meanIntensity .^2)./meanIntensity; % calc contrast for the given pixel along Z (actually the center pixel of the window)
    end
    elapsedTime = toc(startTime);
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'); % delete previous progress line
    fprintf('%05.1f [%%] | %07.1f [sec]', (iX/(lengthX - XYWindowSizePx + 1))*100, ((lengthX - XYWindowSizePx + 1)-iX)*elapsedTime);
end

end

function rtrnXYZLSPContrast = sLASCAConvFilter(InXYZFrames, XYWindowSizePx)
% Conv method (vectorized along Z) to calc the Laser Speckle Contrast

[lengthX, lengthY, lengthZ] = size(InXYZFrames);

% Pre-allocate
rtrnXYZLSPContrast = zeros((lengthX - XYWindowSizePx + 1), (lengthY - XYWindowSizePx + 1), lengthZ);
filterKernel = ones(XYWindowSizePx, XYWindowSizePx); % represents the sliding window

% Calc Laser Speckle Contrast map --> k = std(I)/<I> = sqrt(<I^2> - <I>^2)/<I> = sqrt(<I^2>/<I>^2 - 1)
fprintf('\nProgress LSP Contrast (Conv Filter Calc): 000.0 [%%] | 00000.0 [sec]');
for iZ = 1:lengthZ % loop through image frames
    startTime = tic;
    
    subFrame = InXYZFrames(:, :, iZ); % extract a Z subframe
    % Variant 1
    meanIntensity = filter2(filterKernel, subFrame, 'valid')./XYWindowSizePx^2; % mean intensity along Z
    meanSqrIntensity = filter2(filterKernel, subFrame .^2, 'valid')./XYWindowSizePx^2; % mean of squared intensities along Z
    rtrnXYZLSPContrast(1:end, 1:end, iZ) = sqrt(meanSqrIntensity - meanIntensity .^2)./meanIntensity; % calc contrast for the given pixel along Z (actually the center pixel of the window)
    
    % Variant 2
    %meanIntensity = filter2(filterKernel, subFrame, 'valid')./XYWindowSizePx^2; % mean intensity along Z
    %meanSqrIntensity = filter2(filterKernel, subFrame .^2, 'valid'); % mean of squared intensities along Z
    %sigmaIntensity = sqrt((meanSqrIntensity - XYWindowSizePx^2 .* meanIntensity.^2)./(XYWindowSizePx^2 - 1));
    %rtrnXYZsLSCk(1:end, 1:end, iZ) = sigmaIntensity./meanIntensity; % calc contrast for the given pixel along Z (actually the center pixel of the window)
    
    elapsedTime = toc(startTime);
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'); % delete previous progress line
    fprintf('%05.1f [%%] | %07.1f [sec]', (iZ/lengthZ)*100, (lengthZ - iZ)*elapsedTime);
end

end

function rtrnXYZLSPTc = calcCorrelationTimeMap(LSPContrast, CamExposureTime)
% Calc 3D XYZ Correlation Time Map from Laser Speckle
% Numerical algorithms take ideas from the following papers:
% Julio C. Ramirez-San-Juan et al, "Impact of velocity distribution assumption on simplified laser speckle imaging equation ", DOI link: https://doi.org/10.1364/OE.16.003197
% The algorithm is good approximation for Contrast K in the range, K = [0, 0.6] (T > 2*tc)

fprintf('\nProgress Correlation Time Tc (Calc): ');
startTime = tic;

% Calc correlation time map from contrast map --> we approximate by assuming validity of:
% tc = TK^2 (T = cam exosure time, tc = (de-)correlation time, K = contrast), valid for T > 2*tc
% Then velocity V is --> we use the formula Ct(Tau) = exp(-(Vs*Tau)^2/len0^2) = exp(- Tau^2/Tc^2) => tc = len0/Vs => Vs = len0/Tc
rtrnXYZLSPTc = CamExposureTime .* (LSPContrast .^2); % calc decorrelation time matrix

% Filter velocity with Inf values
% tcthreshold = 5*median(rtrnXYZLSPTc, 'all');
% rtrnXYZLSPTc(isnan(rtrnXYZLSPTc)) = 0;
% rtrnXYZLSPTc(rtrnXYZLSPTc == Inf) = 0;
% rtrnXYZLSPTc(rtrnXYZLSPTc > tcthreshold) = tcthreshold;

% Show progress
elapsedTime = toc(startTime);
fprintf('100%% | %.3f [sec]\n', elapsedTime);
end

function rtrnXYZLSPVelocity = calcVelocityMap(LSPCorrelationTime, WavelengthUm, NA, Magnification)
% Calc 3D XYZ Velocity Map from Laser Speckle
% Numerical algorithms take ideas from the following papers:
% Julio C. Ramirez-San-Juan et al, "Impact of velocity distribution assumption on simplified laser speckle imaging equation ", DOI link: https://doi.org/10.1364/OE.16.003197
% The algorithm is good approximation for Contrast K in the range, K = [0, 0.6] (T > 2*tc)

fprintf('Progress Velocity (Calc): ');
startTime = tic;

% Calc velocity map from contrast map --> we approximate by assuming validity of:
% tc = TK^2 (T = cam exosure time, tc = (de-)correlation time, K = contrast), valid for T > 2*tc
% Then velocity V is --> we use the formula Ct(Tau) = exp(-(Vs*Tau)^2/len0^2) = exp(- Tau^2/Tc^2) => tc = len0/Vs => Vs = len0/Tc
rtrnXYZLSPVelocity = calcTheoryTcToVelocity(LSPCorrelationTime, WavelengthUm, NA, Magnification); % calc velocity map XYZ in [um/s]

% Filter velocity with Inf and/or NaN values
rtrnXYZLSPVelocity(isnan(rtrnXYZLSPVelocity)) = 0;
rtrnXYZLSPVelocity(rtrnXYZLSPVelocity == Inf) = 0;

% Remove velocity outliers
% [lengthX, lengthY, lengthZ] = size(rtrnXYZLSPVelocity);
% for iZ = 1:lengthZ
%     subFrame = rtrnXYZLSPVelocity(:, :, iZ); % get current frame
%     vthreshold = 4*median(subFrame, 'all'); % calc oultlier's upper boundary using the median
%     subFrame(subFrame > vthreshold) = vthreshold;
%     rtrnXYZLSPVelocity(1:end, 1:end, iZ) = subFrame;
% end

% Show progress
elapsedTime = toc(startTime);
fprintf('100%% | %.3f [sec]\n', elapsedTime);
end

function rtrnVs = calcTheoryTcToVelocity(Tc, Wavelength, NA, Magnification)
% Calculate single velocity from Tc (correlation time) and len0 (correlation length)
% Tc = correlation/decorrelation time (where Ct(tau) = 1/e)
% Wavelength = wavelength of the illumination
% NA = numerical aperture of the optical system
% Note: the Vs (velocity) units will depend on Tc unit and Wavelength unit, e.g. if Tc in [s] and Wavelegnth in [um] => Vs in [um/s]

% Calc decorrelation length
len0 = 0.41*Wavelength*Magnification/NA;

% Calc velocity Vs --> we use the formula Ct(Tau) = exp(-(Vs*Tau)^2/len0^2) = exp(- Tau^2/Tc^2) => tc = len0/Vs => Vs = len0/Tc
rtrnVs = len0./Tc;

end

function saveLspData(DataLSP, InputFile, LASCAMethodString, PixelXYZ, XYWindowSizePx, CamExposureTime, NumericalMethod, WavelengthUm, NA, Magnification)
% Save the processed LSP data

fprintf('\nStart saving LSP Data parameters... \n'); % show progress

% Single XY pixel location to calc/show/save curves
pixY = PixelXYZ(1);
pixX = PixelXYZ(2);
pixZ = PixelXYZ(3);
XWindowPx = XYWindowSizePx;
YWindowPx = XYWindowSizePx;

% Options
outputFileType = 'dat';

% Get file name without extension
[inputFilePath, inputFileName, inputFileExtension] = fileparts(InputFile);

% Save common parameters
BaseFileName = [inputFileName LASCAMethodString];
txtFileName = [BaseFileName '.dat'];
fileId = fopen(txtFileName, 'w'); % open the file for writing

% Check if openning file was successful
if (fileId == -1)
    error(['Writing to file failed! --> Filepath = ' txtFileName]);  % inform user about the error
end

% Save parameters --> key = value [unit]
fprintf(fileId, 'XYWindowSizePx = %d [px]\n', XYWindowSizePx);
fprintf(fileId, 'CamExposureTime = %g [s]\n', CamExposureTime);
fprintf(fileId, 'NumericalMethod = %s [-]\n', NumericalMethod);
fprintf(fileId, 'WavelengthUm = %f [um]\n', WavelengthUm);
fprintf(fileId, 'NA = %f [-]\n', NA);
fprintf(fileId, 'Magnification = %f [-]\n', Magnification);
fprintf(fileId, '\n');

% Save LSP Contrast
if ~isempty(DataLSP.Contrast)
    fprintf(fileId, 'Statistics --> Kmax = %f, Kmin = %f, Kmean = %f\n', max(DataLSP.Contrast, [], 'all'), min(DataLSP.Contrast, [], 'all'), mean(DataLSP.Contrast, 'all')); % show progress
    fprintf(fileId, '\n');
end

% Save LSP Correlation Time
if ~isempty(DataLSP.CorrelationTime)
    fprintf(fileId, 'Statistics --> Tcmax = %.3g [s], Tcmin = %.3g [s], Tcmean = %.3g [s]\n', max(DataLSP.CorrelationTime, [], 'all'), min(DataLSP.CorrelationTime, [], 'all'), mean(DataLSP.CorrelationTime, 'all')); % show progress
    fprintf(fileId, '\n');
end

% Save LSP Velocity
if ~isempty(DataLSP.Velocity)
    fprintf(fileId, 'Statistics --> Vmax = %.3f [mm/s], Vmin = %.3f [mm/s], Vmean = %.3f [mm/s]\n', max(DataLSP.Velocity, [], 'all')/1000, min(DataLSP.Velocity, [], 'all')/1000, mean(DataLSP.Velocity, 'all')/1000); % show progress
    fprintf(fileId, '\n');
end

% Statistics in a given pixel
fprintf(fileId, 'Statistics Pixel[%d, %d, %d] --> K = %f, Tc = %g [s], V = %.3f [mm/s]\n', pixY, pixX, pixZ, DataLSP.Contrast(pixX, pixY, pixZ), DataLSP.CorrelationTime(pixX, pixY, pixZ), DataLSP.Velocity(pixX, pixY, pixZ)/1000); % show progress
fprintf(fileId, '\n');

fclose(fileId);

fprintf('End saving LSP Data parameters!\n'); % show progress

end
