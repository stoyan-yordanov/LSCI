function lsci_ConvertToMultiexposureFrames(InputFile, StartFrame, EndFrame, CamExposureTime, MultiExposureAlgorithm, FrameRate, OutputFileType)
% Read input (video or images) with fixed exposure time and Fps = 1/te and convert it to Multiexposure Laser Speckle images/video frames
% Function parameters:
% InputFile = file name of the input data (if empty brings command line file dialog) - supports avi (video) | mj2 (video Motion Jpeg 2000) | tiff (multipage)
% Process between StartFrame and EndFrame frames.
% CamExposureTime = 250e-6 [sec] etc (cam exposure time in sec.)
% MultiExposureAlgorithm = the increase of exposure time with frame number - 'ftime' (new dt = frame time) | 'lin' (new dt = dt + exposure time) | 'pow2' (new dt = 2*dt)
% FrameRate = desired output frames per second
% OutputFileType = 'tiff' (saves as multipage tiff file) | 'avi' (saves as grayscale uncompressed avi movie) | 'mj2' (saves as uncompressed grayscale 16 bits video)


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

% Read input raw (video/image) frames
inXYZFrames = lsci_ReaderFramesToMatrix(InputFile, StartFrame, EndFrame, 'double'); % XY images array (Z = frame index)

% Start processing
fprintf('\nStart Converting to Multiexposure Laser Speckle Frames... \n'); % show progress

% Start Converting
[rows, cols, frames] = size(inXYZFrames);
outXYZFrames = zeros(rows, cols, frames);
outXYFrame = zeros(rows, cols);
frameTime = 1/FrameRate;

% Recalc frames according algorithm --> note: raw frames are assumed to be taken with the same exposure time and with no delay between frames
switch(MultiExposureAlgorithm)
    case 'ftime' % change the exposure time per frame to the one of frame time (by averaging the additional frames), i.e. dt = frame time (i.e. 1/FrameRate)
        i = 0; % output frames counter
        iFrame = 1; % frame counter
        numFramesSubstack = floor(frameTime/CamExposureTime); % number of frames to calc new frames with new exposure time
        
        while iFrame <= (frames - numFramesSubstack + 1)
            i = i + 1;
            outXYFrame = inXYZFrames(:, :, iFrame:(iFrame + numFramesSubstack - 1)); % get substack for the averaging
            outXYZFrames(1:end, 1:end, i) = mean(outXYFrame, 3); % get single frame by averaging over Z direction
            
            % Update frame index
            iFrame = iFrame + numFramesSubstack; % jump to the next block of frames to process
        end
        
        % Remove the empty frames
        outXYZFrames = outXYZFrames(1:end, 1:end, 1:i);
        
    case 'lin' % linear change in exposure time per frame, i.e. dt = dt + CamExposureTime
        i = 0; % output frames counter
        timeStep = 0; % multiexposure time step
        
        for iFrame = 1:frames
            outXYFrame = outXYFrame + inXYZFrames(:, :, iFrame);
            %globalTime = iFrame*CamExposureTime;
            
            % Update multiexposure time step
            timeStep = timeStep + CamExposureTime; % changes in a linear fashion by step = exposure time
            if timeStep > frameTime + 0.1*CamExposureTime % if bigger than frame time + smaller error (avoiding nesty rounding errors)
                % Case the next time step exceed the frame time --> reset
                timeStep = CamExposureTime; % reset multiexposure time
                outXYFrame = inXYZFrames(:, :, iFrame);
            end
            
            i = i + 1;
            outXYZFrames(1:end, 1:end, i) = outXYFrame;
            %outXYZFrames(1:end, 1:end, i) = outXYFrame./max(outXYFrame, [], 'all'); % normalize
        end 
    case 'pow2' % power of 2 change in exposure time per frame, i.e. dt = 2*dt
        i = 0; % output frames counter
        timeStep = CamExposureTime;
        relativeTime = 0;
        
        for iFrame = 1:frames
            outXYFrame = outXYFrame + inXYZFrames(:, :, iFrame); % accumulate frames
            relativeTime = relativeTime + CamExposureTime; % tracks time progression between frames
                                   
            % Update multiexposure time step
            if timeStep > frameTime + 0.1*CamExposureTime % if bigger than frame time + smaller error (avoiding nesty rounding errors)
                % Case the next time step exceed the frame time --> reset
                timeStep = CamExposureTime; % reset multiexposure time
                relativeTime = CamExposureTime;
                outXYFrame = inXYZFrames(:, :, iFrame); % reset frame
            end
            
            % Update pow2 counter time step
            if relativeTime > (timeStep - 0.1*CamExposureTime) && relativeTime < (timeStep + 0.1*CamExposureTime) % if bigger than frame time + smaller error (avoiding nesty rounding errors)
                i = i + 1; % pow2 counter
                outXYZFrames(1:end, 1:end, i) = outXYFrame;
                %outXYZFrames(1:end, 1:end, i) = outXYFrame./max(outXYFrame, [], 'all'); % normalize 
                timeStep = 2*timeStep; % changes in a power of 2 fashion
            end            
        end
        
        % Reduce size of the 3D stack to the actual multiexposure frames
        if frames > 1 && i > 1
            outXYZFrames = outXYZFrames(:, :, 1:i);
        end
    otherwise
        fprintf('\n\nYou have chosen unsupported multiexposure algorithm --> Multiexposure Algorithm = %s\n', MultiExposureAlgorithm);
        error('Exit due to the error above!');
end


% Build file name
[outRows, outCols, outFrames] = size(outXYZFrames);
strDelimiter = '_';
fileNameStrCellArray = splitFileNameByDelimiter(fileName, strDelimiter); % split file name in strings by delimiter

xytzString = sprintf('xytz=%dx%dx%d', outCols, outRows, outFrames);
[fileNameStrCellArray, isPatternFound] = replaceStrPartsInStrCellArray(fileNameStrCellArray, 'xytz=', xytzString);
if ~isPatternFound
    fileNameStrCellArray{end+1} = xytzString;
end

meString = sprintf('me=%s', MultiExposureAlgorithm);
[fileNameStrCellArray, isPatternFound] = replaceStrPartsInStrCellArray(fileNameStrCellArray, 'me=', meString);
if ~isPatternFound
    fileNameStrCellArray{end+1} = meString;
end

fpsString = sprintf('fps=%d(on)', FrameRate);
[fileNameStrCellArray, isPatternFound] = replaceStrPartsInStrCellArray(fileNameStrCellArray, 'fps=', fpsString);
if ~isPatternFound
    fileNameStrCellArray{end+1} = fpsString;
end

baseFileName = buildStringFromCellArray(fileNameStrCellArray, strDelimiter);

% Save processed data as tiff or video
type3DStackItNormalization = 'global';
lsci_SaveToFrames(outXYZFrames, baseFileName, OutputFileType, type3DStackItNormalization);

% Print progress
elapsedTime = toc(startTime);

fprintf('\n\nEnd of processing --> Start Frame = %d, End Frame = %d\n', StartFrame, EndFrame); % show progress
fprintf('Stat of processing --> %s, %s, %s\n', xytzString, meString, fpsString); % show progress
fprintf('Output file --> %s\n', [baseFileName '.' OutputFileType]); % show progress
fprintf('Processing time = %f [sec]\n\n', elapsedTime);

end

function rtrnStrCellArray = splitFileNameByDelimiter(FileName, StrDelimiter)
% Strings in the file name get splitted by delimiter

rtrnStrCellArray = textscan(FileName, '%s', 'Delimiter', StrDelimiter);
rtrnStrCellArray = rtrnStrCellArray{:};

end

function [rtrnStrCellArray, rtrnIsPatternFound] = replaceStrPartsInStrCellArray(InputStrCellArray, StrPattern, NewString)
% Replace strings in the cell array

rtrnIsPatternFound = false;

for i = 1:size(InputStrCellArray, 1)
    if contains(InputStrCellArray{i}, StrPattern)
        InputStrCellArray{i} = NewString;
        rtrnIsPatternFound = true;
        break;
    end
end
    
rtrnStrCellArray = InputStrCellArray;

end

function rtrnString = buildStringFromCellArray(InputStrCellArray, StrDelimiter)
% Build string from cell array elements

rtrnString = '';

for i = 1:size(InputStrCellArray, 1)
    rtrnString = [rtrnString InputStrCellArray{i}]; % add string
    
    if i < size(InputStrCellArray, 1)
        rtrnString = [rtrnString StrDelimiter]; % add delimiter to string
    end
end

end
