function lsci_teLASEA(InputFile, StartFrame, EndFrame, EntropyEstimator, PixelXYZ, ZWindowSizePx, BitsDepth, CamExposureTime)
% Read video, calc Temporal Laser Speckle Entropy (LSE) (in temporal domain) and save it as multi page tiff file.
% InputFile = file name of the input data (if empty brings command line file dialog) - supports avi (video) | mj2 (video Motion Jpeg 2000) | tiff (multipage)
% Process between StartFrame and EndFrame frames.
% EntropyEstimator = 'Miller' | 'Balanced' (Miller = improved naive entropy estimator for many frames; Balanced = balanced entropy estimator for low frame number, e.g. below 100)
% PixelXYZ = [X, Y, Z] (coordinate of the point where we show statistics (H and V)
% ZWindowSizePx = 25, 50, 100 etc (pixel size of the Z sliding window to calc entropy per pixel)
% BitDepth = 8 | 16 bits data values
% CamExposureTime = 250e-6 [sec] etc (cam exposure time in sec.)

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
switch(BitsDepth)
    case 8
        inXYZFrames = lsci_ReaderFramesToMatrix(InputFile, StartFrame, EndFrame, 'uint8'); % convert to uint8
    case 16
        inXYZFrames = lsci_ReaderFramesToMatrix(InputFile, StartFrame, EndFrame, 'uint16'); % convert to uint16        
    otherwise
        frintf('\n\nUnsupported bits depth  --> Bits Depth = %d\n', BitsDepth);
        error('Exit due to error!');
end

% The structure that will store the results of the processing
dataLSP = struct();
dataLSP.Entropy = []; % LSP Entropy --> represents H
dataLSP.Velocity = []; % Velocity --> represents V

% Write frames to multipage tiff file
fprintf('\nStart calculating Laser Speckle Entropy... \n'); % show progress

% Process input and calc Laser Speckle Entropy Map
dataLSP.Entropy = lsciProcessLASEAInputByMethod(inXYZFrames, EntropyEstimator, ZWindowSizePx, BitsDepth);

% Process Laser Speckle Entropy Map and calc Velocity Map
dataLSP.Velocity = calcVelocityMap(dataLSP.Entropy, ZWindowSizePx, BitsDepth, CamExposureTime);

% Save result for Laser Speckle Entropy as tiff
tiffFileNamePathLSPEntropy = fullfile(filePath, [fileName '_teLSE-h' '.tiff']); % Assemble tiff file name for Laser Speckle Entropy
tiffFileNamePathLSPVelocity = fullfile(filePath, [fileName '_teLSE-v' '.tiff']); % Assemble tiff file name for Laser Speckle Velocity
[heightX, widthY, depthZ] = size(dataLSP.Entropy);
for i = 1:depthZ
        
    % Prepare for global normalization
    if i == 1
        minH = min(dataLSP.Entropy, [], 'all');
        maxH = max(dataLSP.Entropy, [], 'all');
        minV = min(dataLSP.Velocity, [], 'all');
        maxV = max(dataLSP.Velocity, [], 'all');
    end
    
    % Get current frame
    frameH = (dataLSP.Entropy(:, :, i));
    frameV = (dataLSP.Velocity(:, :, i));
        
    frameH = (frameH - minH)./(maxH - minH); % normalize to max. 1    
    frameV = (frameV - minV)./(maxV - minV); % normalize to max. 1
    
    %cInt8 = 255; % coefficient to convert to 8 bit integer
    %frameH = uint8(cInt8*frameH); % convert to 8 bit depth
    %frameV = uint8(cInt8*frameV); % convert to 8 bit depth
    
    cInt16 = 2^16 - 1; % coefficient to convert to 16 bit integer
    frameH = uint16(cInt16*frameH); % convert to 16 bit depth
    frameV = uint16(cInt16*frameV); % convert to 16 bit depth
    
    imwrite(frameH, tiffFileNamePathLSPEntropy, 'tiff', 'Compression', 'packbits', 'WriteMode', 'append'); % save Laser Speckle Entropy
    imwrite(frameV, tiffFileNamePathLSPVelocity, 'tiff', 'Compression', 'packbits', 'WriteMode', 'append'); % save Laser Speckle Velocity
end

% Save processed data
saveLspData(dataLSP, InputFile, '_teLSE', PixelXYZ, ZWindowSizePx, CamExposureTime, EntropyEstimator, BitsDepth);

% Print progress
elapsedTime = toc(startTime);

fprintf('\n\nEnd of processing --> Start Frame = %d, End Frame = %d\n', StartFrame, EndFrame); % show progress
fprintf('Entropy statistics --> Global: minH = %.5f, maxH = %.5f\n', minH, maxH); % show entropy stat
fprintf('Entropy statistics pixel[%d, %d, %d] --> H = %.5f, V = %.3f [mm/s]\n', pixY, pixX, pixZ, dataLSP.Entropy(pixY, pixX, pixZ), dataLSP.Velocity(pixY, pixX, pixZ)); % show entropy stat
fprintf('Processing time = %f [sec]\n\n', elapsedTime);

end

function rtrnXYZLSPEntropy = lsciProcessLASEAInputByMethod(InXYZFrames, EntropyEstimator, ZWindowSizePx, BitsDepth)
% Calc Laser Speckle Entropy Map
% The Laser Speckle Entropy algorithm ('Balanced' estimator) is implemented based on the following paper (see eq. 4 and 5):
% Peng Miao et al, "Entropy analysis reveals a simple linear relation", DOI link: http://dx.doi.org/10.1364/OL.39.003907
% The Laser Speckle Entropy algorithm ('Miller' or improved naive estimator) is implemented based on the following paper (see eq. 8 and 12):
% Juan A Bonachela et al, "Entropy estimates of small data sets", DOI link: http://dx.doi.org/10.1088/1751-8113/41/20/202001

% Calc Laser Speckle Entropy map
switch(EntropyEstimator)
    case 'Miller'
        rtrnXYZLSPEntropy = eLASEAMiller(InXYZFrames, ZWindowSizePx, BitsDepth);
    case 'Balanced'
        rtrnXYZLSPEntropy = eLASEASBalanced(InXYZFrames, ZWindowSizePx, BitsDepth);
    otherwise
        frintf('\n\nUnsupported entropy estimator method --> %s\n', EntropyEstimator);
        error('Exit due to error!');
end

end

function rtrnXYZLSPVelocity = calcVelocityMap(DataLSPEntropy, ZWindowSizePx, BitsDepth, CamExposureTime)
% Calc Laser Speckle Velocity Map
% The Laser Speckle Entropy algorithm ('Balanced' estimator) is implemented based on the following paper (see eq. 4 and 5):
% Peng Miao et al, "Entropy analysis reveals a simple linear relation", DOI link: http://dx.doi.org/10.1364/OL.39.003907
% The Laser Speckle Entropy algorithm ('Miller' or improved naive estimator) is implemented based on the following paper (see eq. 8 and 12):
% Juan A Bonachela et al, "Entropy estimates of small data sets", DOI link: http://dx.doi.org/10.1088/1751-8113/41/20/202001

fprintf('\nProgress Velocity (Calc): ');
startTime = tic;

% Calc velocity map from entropy map --> v = (c + a*T - H)/b
%rtrnXYZLSPVelocity = (a*CamExposureTime + c - dataLSPEntropy)/b;
a = 1.29e+5; % exposure time coefficient - needs to be defined experimentally
b = 100; % velocity coefficient - needs to be defined experimentally
c = entropyShannonBalancedH0(2^BitsDepth, ZWindowSizePx); % calc entropy constant 'c' for the velocity estimation

rtrnXYZLSPVelocity = (c + a*CamExposureTime - DataLSPEntropy)./b; % in [mm/s]

% Filter velocity with Inf values
rtrnXYZLSPVelocity(isnan(rtrnXYZLSPVelocity)) = 0;
rtrnXYZLSPVelocity(rtrnXYZLSPVelocity == Inf) = 0;

% Show progress
elapsedTime = toc(startTime);
fprintf('100 [%%] | %.3f [sec]', elapsedTime);

end

function rtrnXYZeLSEh = eLASEAMiller(InXYZFrames, ZWindowSizePx, BitsDepth)
% Calc of Laser Speckle Entropy using Miller Entropy Estimator

[lengthX, lengthY, lengthZ] = size(InXYZFrames);

% Pre-allocate
rtrnXYZeLSEh = zeros(lengthX , lengthY, lengthZ - ZWindowSizePx + 1);

% Counter of states 2^bits, 8 bits --> M = 256 states, 16 bits --> M = ?65536? states
ncounter = zeros(1, 2^BitsDepth); % must be kept always with zeros to have proper init below

% Calc Laser Speckle Entropy map --> H = sum_(i=1:M)(-n_i/N*ln(n_i/N) + 1/2N)
fprintf('\nProgress LSP Entropy (Miller): 000.0 [%%] | 00000.0 [sec]');
%workerThreads = 4;
%parfor (iZ = 1:(lengthX - ZWindowSizePx), workerThreads) % parallel for-loop through frames
for iZ = 1:(lengthZ - ZWindowSizePx + 1) % loop thorough frames
    startTime = tic;
    % Calc Laser Speckle Entropy for XY frame taking the Z-window
    for iX = 1:lengthX % loop through image height
        for iY = 1:lengthY % loop through image width
            subFrame = InXYZFrames(iX, iY, iZ:(iZ + ZWindowSizePx - 1)); % extract subframe given by the window size
            nicounts = freqCounter(ncounter, subFrame); % return freq occurences for each gray level state M            
            rtrnXYZeLSEh(iX, iY, iZ) = entropyShannonMiller(nicounts, ZWindowSizePx); % calc entropy for the given pixel
        end
    end
    elapsedTime = toc(startTime);
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'); % delete previous progress line
    fprintf('%05.1f [%%] | %07.1f [sec]', (iZ/(lengthZ - ZWindowSizePx + 1))*100, ((lengthZ - ZWindowSizePx + 1) - iZ)*elapsedTime);
end

end

function rtrnXYZeLSEh = eLASEASBalanced(InXYZFrames, ZWindowSizePx, BitsDepth)
% Calc of Laser Speckle Entropy using Balanced Entropy Estimator

[lengthX, lengthY, lengthZ] = size(InXYZFrames);

% Pre-allocate
rtrnXYZeLSEh = zeros(lengthX , lengthY, lengthZ - ZWindowSizePx + 1);

% Counter of states 2^bits, 8 bits --> M = 256 states, 16 bits --> M = ?65536? states
ncounter = zeros(1, 2^BitsDepth); % must be kept always with zeros to have proper init below

% Calc Laser Speckle Entropy map --> H = 1/(N+2) * sum_(i=1:M)((n_i + 1) sum_(j=n_i+2:N+2)(1/j))
fprintf('\nProgress LSP Entropy (Balanced): 000.0 [%%] | 00000.0 [sec]');
%workerThreads = 4;
%parfor (iZ = 1:(lengthX - ZWindowSizePx + 1), workerThreads) % parallel for-loop through frames
for iZ = 1:(lengthZ - ZWindowSizePx + 1) % loop thorough frames
    startTime = tic;
    % Calc Laser Speckle Entropy for XY frame taking the Z-window
    for iX = 1:lengthX % loop through image height
        for iY = 1:lengthY % loop through image width
            subFrame = InXYZFrames(iX, iY, iZ:(iZ + ZWindowSizePx - 1)); % extract subframe given by the window size
            nicounts = freqCounter(ncounter, subFrame); % return freq occurences for each gray level state M
            rtrnXYZeLSEh(iX, iY, iZ) = entropyShannonBalanced(nicounts, ZWindowSizePx); % calc entropy for the given pixel
        end
    end
    elapsedTime = toc(startTime);
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'); % delete previous progress line
    fprintf('%05.1f [%%] | %07.1f [sec]', (iZ/(lengthZ - ZWindowSizePx + 1))*100, ((lengthZ - ZWindowSizePx + 1) - iZ)*elapsedTime);
end

end

function [NCounter] = freqCounter(NCounter, InputArray)
% Frequency count the number of occurences of the input array elements (positive integers expected)
% InputArray = array of elements (it can be multi dim array), positive integers
% NCounter = one dim array with the integer numbers frequency (note: index num+1 stores freq, occur. for num)

% Calc entropy --> H = sum_(i=1:M)(-n_i/N*ln(n_i/N) + 1/2N)
%inputVector = InputArray(:); % get all elements in linear one dim vector

for i = 1:numel(InputArray) % use the linear indexing of multi array
    num = InputArray(i);
    NCounter(num + 1) = NCounter(num + 1) + 1; % incease counter of num (index num+1) with one
end

end

function rtrnH = entropyShannonMiller(NiArray, N)
% Entropy calcculated using Shannon entropy as corrected by Miller (see the respective paper, eq. 8 and eq. 12)
% The Laser Speckle Entropy algorithm ('Miller' or improved naive estimator) is implemented based on the following paper (see eq. 8 and 12):
% Juan A Bonachela et al, "Entropy estimates of small data sets", DOI link: http://dx.doi.org/10.1088/1751-8113/41/20/202001

rtrnH = 0; % init entropy
piArray = NiArray ./ N; % calc probabilities of occurencies of each element
for i = 1:length(NiArray)
    if piArray(i) == 0
        rtrnH = rtrnH + 1/(2*N);
    else
        rtrnH = rtrnH + (- piArray(i)*log(piArray(i)) + 1/(2*N));
    end
end

%rtrnH = sum(- piArray.*log(piArray) + 1/(2*N));

end

function rtrnH = entropyShannonBalanced(NiArray, N)
% Entropy calcculated using Shannon entropy and the 'balanced' entropy estimator (see eq. 4 in the paper below, this is what we calculate here)
% Peng Miao et al, "Entropy analysis reveals a simple linear relation", DOI link: http://dx.doi.org/10.1364/OL.39.003907
% NiArray = array of freq occurencies of each num = i-1 of NiArray() --> NiArray(i) gives the occurency count of i-1 number
% N = size of the data set where the freq counting was performed (note: N is different than the size of NiArray)

% Calc entropy --> H = 1/(N+2) * sum_(i=1:M)((n_i + 1) sum_(j=n_i+2:N+2)(1/j))
rtrnH = 0; % init entropy
% piArray = (NiArray + 1)./ (N+2); % calc balanced estimator probabilities of occurencies of each element
% for i = 1:length(NiArray)
%     % Calc most internal sum along j
%     sumj = 0;
%     for j = (NiArray(i)+2):(N+2)
%         sumj = sumj + 1/j;
%     end
%     rtrnH = rtrnH + piArray(i) * sumj;
% end

for i = 1:length(NiArray)
    % Calc most internal sum along j
    sumj = 0;
    for j = (NiArray(i)+2):(N+2)
        sumj = sumj + 1/j;
    end
    rtrnH = rtrnH + (NiArray(i) + 1) * sumj;
end

rtrnH = rtrnH/(N+2);

end

function rtrnH0 = entropyShannonBalancedH0(M, N)
% Calcs the entropy constant 'c' (when T=0 and v = 0) in the equation used to extract the velocity (for more details see 'c' constant is eq. 5 in the paper below)
% Peng Miao et al, "Entropy analysis reveals a simple linear relation", DOI link: http://dx.doi.org/10.1364/OL.39.003907
% M = number of states (num possible gray levell values, e.g. 2^8 or 2^16)
% N = size of the dataset

% Calc entropy --> H0 (or 'c') = (N+1)/(N+2)^2 + ((M-1)/(N+2))*sum_(j=2:N+2)(1/j)
rtrnH0 = (N+1)/(N+2)^2; % first term in the H0 entropy above
sumj = 0;
for j = 2:(N+2)
    % Calc most internal sum (along j)
    sumj = sumj + 1/j;
end

rtrnH0 = rtrnH0 + ((M-1)/(N+2))*sumj;

% Alternative (but much slower, one line) way to calc. using psi(x) (digamma function) and eulergamma constant (=0.5772156649...)
%rtrnH0 = (N+1)/(N+2)^2 + ((M-1)/(N+2))*(psi(N+3) - 1 + 0.5772156649);

end

function saveLspData(DataLSP, InputFile, LASCAMethodString, PixelXYZ, ZWindowSizePx, CamExposureTime, EntropyEstimator, BitsDepth)
% Save the processed LSP data

fprintf('\nStart saving LSP Data parameters... \n'); % show progress

% Single XY pixel location to calc/show/save curves
pixY = PixelXYZ(1);
pixX = PixelXYZ(2);
pixZ = PixelXYZ(3);

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
fprintf(fileId, 'ZWindowSizePx = %d [px]\n', ZWindowSizePx);
fprintf(fileId, 'CamExposureTime = %g [s]\n', CamExposureTime);
fprintf(fileId, 'EntropyEstimator = %s [-]\n', EntropyEstimator);
fprintf(fileId, 'BitsDepth = %d [-]\n', BitsDepth);
fprintf(fileId, '\n');

% Save LSP Entropy
if ~isempty(DataLSP.Entropy)
    fprintf(fileId, 'Statistics --> Hmax = %f, Hmin = %f, Hmean = %f\n', max(DataLSP.Entropy, [], 'all'), min(DataLSP.Entropy, [], 'all'), mean(DataLSP.Entropy, 'all')); % show progress
    fprintf(fileId, '\n');
end

% Save LSP Velocity
if ~isempty(DataLSP.Velocity)
    fprintf(fileId, 'Statistics --> Vmax = %.3f [mm/s], Vmin = %.3f [mm/s], Vmean = %.3f [mm/s]\n', max(DataLSP.Velocity, [], 'all'), min(DataLSP.Velocity, [], 'all'), mean(DataLSP.Velocity, 'all')); % show progress
    fprintf(fileId, '\n');
end

% Statistics in a given pixel
fprintf(fileId, 'Statistics Pixel[%d, %d, %d] --> H = %f, V = %.3f [mm/s]\n', pixY, pixX, pixZ, DataLSP.Entropy(pixX, pixY, pixZ), DataLSP.Velocity(pixX, pixY, pixZ)); % show progress
fprintf(fileId, '\n');

fclose(fileId);

fprintf('End saving LSP Data parameters!\n'); % show progress

end