function lsci_VideosToMultiTiff(InputFile, StartFrame, EndFrame, FileTypeFilterString)
% Read and convert videos to image (multi page tiff) files

% Case no file provided or the string is not a valid file --> Get dir file list and choose a file t process
if strcmp(InputFile, '') || ~isfile(InputFile)
    fileList = lsci_sysGetDirectoryFileList(FileTypeFilterString); % return the list of file in the current dir
    fileList = lsci_sysChooseFilesFromFileList(fileList); % get the file(s) to be processed    
else
    fileList{1, 1} = InputFile; % single input file
end

% Get number of files
lengthFileList = size(fileList, 1);

startTime = tic;

% Loop through files
for i = 1:lengthFileList
    % Write frames to multipage tiff file
    fprintf('\nStart frames to multi-page tiff conversion of file #%d... ', i); % show progress
    startTime2 = tic;
    
    % Get current input file
    InputFile = fileList{i, 1}; % only one file (the first one) will be processed
    
    % Check file name
    [filePath, fileName, fileExtension] = fileparts(InputFile);
    
    % Read input raw frames
    inXYZFrames = lsci_ReaderFramesToMatrix(InputFile, StartFrame, EndFrame, 'double'); % XY images array (Z = frame index)
    
    % Assemble tiff file name
    tiffBaseFileNamePath = fullfile(filePath, fileName);
            
    % Save resulting file as multipage tiff
    type3DStackItNormalization = 'global';
    outputFileType = 'tiff';
    lsci_SaveToFrames(inXYZFrames, tiffBaseFileNamePath, outputFileType, type3DStackItNormalization);
    
    % Show progress
    elapsedTime2 = toc(startTime2);
    
    fprintf('\nEnd of conversion of file #%d --> Start Frame = %d, End Frame = %d\n', i, StartFrame, EndFrame); % show progress
    fprintf('\nWriting to output file took:\n --> %.3f [sec]\n --> %s\n', elapsedTime2, tiffBaseFileNamePath);    
    
    fprintf('\nProgress: %.1f [%%] | %.3f [sec]\n', (i/lengthFileList)*100, (lengthFileList - i)*elapsedTime2);
end

fprintf('\nOverall processing time: %.3f [sec]\n', toc(startTime));

end

