function lsci_SaveToFrames(OutputXYZFrames, BaseFileName, OutputFileType, NormalizationType)
% Save images/video frames
% Function parameters:
% OutputXYZFrames = output data frames to be saved
% BaseFileName = file name of the output data (without the extension)
% OutputFileType = 'tiff' (saves as multipage tiff file) | 'avi' (saves as grayscale uncompressed avi movie) | 'mj2' (saves as uncompressed grayscale 16 bits video)
% NormalizationType = 'local' (takes the max intensity in a frame) | 'global' (takes the max intensity in the 3D stack)
% ScalingType = 'max' (scales data to the max found value) | 'min-max' (scales data between min-max found values)

startTime = tic;

% Start processing
fprintf('\nStart Saving Frames...'); % show progress

% Calc normalization factor
maxIt = zeros(1, size(OutputXYZFrames, 3));
switch(NormalizationType)
    case 'local'
        
        % Calc normalization loclly, i.e. for every frame        
        for iZ = 1:size(OutputXYZFrames, 3)            
            maxIt(1) = max(OutputXYZFrames(:, :, iZ), [], 'all'); % calc local normalization factor per frame            
        end
        
    case 'global'
        
        % Calc normalization globally
        maxIt(1:end) = max(OutputXYZFrames, [], 'all'); % global noramlization factor is the same for each frame
        
    otherwise
      fprintf('\n\nUnsupported normalization type --> Normalization Type = %s\n', NormalizationType);
      error('Exit due to the error above!');
end

% Save processed data as tiff or video
switch(OutputFileType)
    case {'tiff', 'tif'}
        fileNameAndPath = fullfile([BaseFileName '.tiff']); % Assemble tiff file name        
        
        % Variant 1 --> uses tiff lib from Matlab
        for iZ = 1:size(OutputXYZFrames, 3)
                        
            frameLSP = OutputXYZFrames(:, :, iZ)./maxIt(iZ); % normalize to 1
            cInt = 2^16 - 1; % cInt = 255; % coefficient to convert to 8 or 16 bit integer
            frameLSP = uint16(cInt.*frameLSP); % convert to 8 or 16 bit depth
                                  
            imwrite(frameLSP, fileNameAndPath, 'tiff', 'Compression', 'packbits', 'WriteMode', 'append'); % save Laser Speckle 3D stack            
        end
        
        % Variant 2 --> uses custom tiff lib
%         objFastTiffWiter = TiffWriterClass(fileNameAndPath, 37.8, 0); % object to tiff file        
%         for iZ = 1:size(OutputXYZFrames, 3)
%                         
%             frameLSP = OutputXYZFrames(:, :, iZ)./maxIt(iZ); % normalize to 1
%             cInt = 2^16 - 1; % cInt = 255; % coefficient to convert to 8 or 16 bit integer
%             frameLSP = uint16(cInt.*frameLSP); % convert to 8 or 16 bit depth
%                         
%             objFastTiffWiter.WriteImage(frameLSP); % save Laser Speckle 3D stack                      
%         end
%         objFastTiffWiter.close; % close tiff file
        
    case 'avi'
        fileNameAndPath = fullfile([BaseFileName '.avi']); % Assemble tiff file name
        
        videoObject = VideoWriter(fileNameAndPath, 'Grayscale AVI'); % Uncompressed AVI file with grayscale video
        %videoObject = VideoWriter(fileNameAndPath, 'Uncompressed AVI'); % Uncompressed AVI file with RGB24 video
        videoObject.FrameRate = 30;
        open(videoObject);
        
        for iZ = 1:size(OutputXYZFrames, 3)
                        
            frameLSP = OutputXYZFrames(:, :, iZ)./maxIt(iZ); % normalize to 1
            cInt = 255; % coefficient to convert to 8 bit integer
            frameLSP = uint8(cInt.*frameLSP); % convert to 8 bit depth
            
            writeVideo(videoObject, frameLSP); % save Laser Speckle 3D stack
        end
        
        close(videoObject);
    case 'mj2' % Motion JPEG 2000 video - it can support gray + color videos with 8 and 16 bits depths (mj2 can be played with ffplay from ffmpeg software)
        fileNameAndPath = fullfile([BaseFileName '.mj2']); % Assemble tiff file name
        
        videoObject = VideoWriter(fileNameAndPath, 'Archival'); % Motion JPEG 2000 file with lossless compression
        videoObject.FrameRate = 30;
        videoObject.MJ2BitDepth = 16; % set 16 bits depth
        open(videoObject);
        
        for iZ = 1:size(OutputXYZFrames, 3)
            
            frameLSP = OutputXYZFrames(:, :, iZ)./maxIt(iZ); % normalize to 1
            cInt = 2^16 - 1; % coefficient to convert to 8 bit integer
            frameLSP = uint16(cInt.*frameLSP); % convert to 8 bit depth
            
            writeVideo(videoObject, frameLSP); % save Laser Speckle 3D stack
        end
        
        close(videoObject);
    otherwise
        fprintf('\n\nUnsupported file type --> File Type = %s\n', OutputFileType);
        error('Exit due to the error above!');
end

% Print progress
elapsedTime = toc(startTime);

fprintf('\nEnd saving file --> Output file: %s\n', fileNameAndPath); % show progress
fprintf('Processing time = %f [sec]\n', elapsedTime);

end
