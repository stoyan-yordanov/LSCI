function rtrnMatrix = lsci_ReaderFramesToMatrix(InputFile, StartFrame, EndFrame, OutputDataFormat)
% Read input file (video/images in grayscale format), convert it to matrix (XxYxZ, Z = frame index number) form suitable for further processing
% InputFile = file name of the input data (video or images) - supports avi (video) | mj2 (video Motion JPEG 2000) | tiff (multipage)
% StartFrame = the starting frame (might not be the first one)
% EndFrame = the end frame (use number to read till specific frame or 'Inf' to read till the end of the file)
% OutputDataFormat = 'double' | 'uint8' (8 bits depth) | uint16 (16 bits depth)

startTime = tic;
fprintf('\n\nLoad input file:\n --> %s\n', InputFile);

% Check file name
[filePath, fileName, fileExtension] = fileparts(InputFile);

% Check if file exists
if ~isfile(InputFile)
    fprintf('\n\nProblem reading input file: %s\n', [fileName fileExtension]);
    error('The file does not exists!');
end

% Read input file
switch(fileExtension)
    case {'.avi', '.mj2'}
        % Read frames
        videoObj = VideoReader(InputFile); % create video object
        
        % Initial preallocation of the matrix (including color video)
        %inputFrames = zeros(videoObj.Height, videoObj.Width, 3, 100); % a frame buffer for 100 images
        
        % Get number of frames in the file
        numberOfFrames = 0;
        while hasFrame(videoObj)
            numberOfFrames = numberOfFrames + 1;
            readFrame(videoObj);
            %inputFrames(1:end, 1:end, 1:end, numberOfFrames) = readFrame(videoObj);
        end
        
        % Re-create video object in order to read the video file
        videoObj = VideoReader(InputFile); 
        
        % Print stat for file
        fprintf('\nVideo properties: \n');
        fprintf(' --> Size = %dx%d, Duration = %.3f [sec], Frame Rate = %d, BitsPerPixel = %d, VideoFormat = %s\n',...
            videoObj.Width, videoObj.Height, videoObj.Duration, videoObj.FrameRate, videoObj.BitsPerPixel, videoObj.VideoFormat);
                               
        % Set start and end frames if empty or not valid
        [StartFrame, EndFrame] = setStartEndFrames(StartFrame, EndFrame, numberOfFrames);
        
        % Read frames between the start and end indexes        
        rtrnMatrix = read(videoObj, [StartFrame EndFrame]); % read subframe stack between start frame and end frame
        %rtrnMatrix = inputFrames(1:end, 1:end, 1:end, StartFrame:EndFrame); % get subframe stack between start frame and end frame
                
        % Convert to matrix
        % z = 0;
        %for i = StartFrame:EndFrame
        %   z = z + 1;
            % Frames representation --> (Height/Row, Width/Column, ColorChannel, FrameIndex)
        %   frame = inputFrames(:, :, 1, i); % extract the current frame and present it in grayscale (third index 1 to select the first color channel)
        %   rtrnMatrix(1:end, 1:end, z) = frame(:, :, 1, 1);
        %end

        % Convert to matrix
        switch(videoObj.VideoFormat)
            case {'RGB24', 'RGB24 Signed', 'RGB48', 'RGB48 Signed'} % three color channels video
                
                % Case of color video -->  XxYxChxZ, Ch = 3 - we reduce it to single channel video
                rtrnMatrix = permute(rtrnMatrix, [1, 2, 4, 3]); % we take only the third color channel as a gray scale value
                rtrnMatrix = rtrnMatrix(:, :, :, 1); % reduce dimensions to 3
                
            case {'Indexed', 'Grayscale', 'Mono8', 'Mono8 Signed', 'Mono16', 'Mono16 Signed'} % mono video format
                
                % Case of mono video --> XxYxChxZ, Ch = 1
                rtrnMatrix = permute(rtrnMatrix, [1, 2, 4, 3]); % permute to reduce dimensions to 3
                %rtrnMatrix = rtrnMatrix(:, :, :, 1); % reduce dimensions to 3
                
            otherwise
                fprintf('\nThe video format in the input file is not supported --> Vidoe Foramt = %s\n', videoObj.VideoFormat);
                error('Exit due to the error above!');
        end        
    case {'.tiff', '.tif'} % case of multipage tiff/tiff image files
        % Get file info
        imInfo = imfinfo(InputFile);
        numImages = numel(imInfo); % number of images
        
        % Print stat for file
        fprintf('\nImages properties: \n');
        fprintf(' --> Size = %dx%dx%d, BitDepth = %d, Format = %s\n',...
            imInfo(1).Width, imInfo(1).Height, numImages, imInfo(1).BitDepth, imInfo(1).ColorType);
        
        % Set start and end frames if empty or not valid
        [StartFrame, EndFrame] = setStartEndFrames(StartFrame, EndFrame, numImages);
        
        % Preallocate the matrix
        rtrnMatrix = zeros(imInfo(1).Height, imInfo(1).Width, EndFrame - StartFrame + 1);
                
        % Convert to matrix
        switch(imInfo(1).ColorType)
            case {'truecolor'} % three color channels image               
                
                % Read multipage image file                
                z = 0;
                for i = StartFrame:EndFrame
                    z = z + 1;
                    % Frames representation --> (Height/Row, Width/Column, ColorChannel, FrameIndex)
                    frame = imread(InputFile, i); % read image with the given index i
                    rtrnMatrix(1:end, 1:end, z) = frame(:, :, 1); % convert to gray scale                    
                end
                
            case {'indexed', 'grayscale'} % mono image format
                
                % Read multipage image file
                z = 0;
                for i = StartFrame:EndFrame
                    z = z + 1;
                    % Frames representation --> (Height/Row, Width/Column, ColorChannel, FrameIndex)
                    frame = imread(InputFile, i); % read image with the given index i
                    rtrnMatrix(1:end, 1:end, z) = frame;                  
                end
                
            otherwise
                fprintf('\nThe image format of the input file is not supported --> Image Foramt = %s\n', imInfo(1).ColorType);
                error('Exit due to the error above!');
        end
        
    otherwise
        fprintf('\n\nUnsupported file type --> File Type = %s\n', fileExtension);
        error('Exit due to the error above!');
end

% Convert to specified data type
rtrnMatrix = setMatrixDataFormat(rtrnMatrix, OutputDataFormat);

% Print progress
fprintf('\nInput file loading time = %.3f [sec]\n', toc(startTime));

end

function InputMatrix = setMatrixDataFormat(InputMatrix, OutputDataFormat)
% Convert to the given output foramt

switch(OutputDataFormat)
    case 'uint8'
        cInt8 = 255; % coefficient to convert to 8 bit integer
        InputMatrix = cInt8.*(InputMatrix./max(InputMatrix, [], 'all'));
        InputMatrix = uint8(InputMatrix);
    case 'uint16'
        InputMatrix = uint16(InputMatrix);
    case 'double'
        InputMatrix = double(InputMatrix);
    otherwise
        fprintf('\n\nUnsupported output data type --> Data Type = %s\n', OutputDataFormat);
        error('Exit due to the error above!');
end

end

function [StartFrame, EndFrame] = setStartEndFrames(StartFrame, EndFrame, Frames)
% Check validity and set start and end frames

if EndFrame < 1 || EndFrame > Frames
    EndFrame = Frames;
end

if StartFrame < 1 || StartFrame > Frames
    StartFrame = 1;
end

if StartFrame > EndFrame
    fprintf('\nSatrt frame index is bigger than End frame index --> StartFrame = %d, EndFrame = %d\n', StartFrame, EndFrame);
    error('Exit due to above error');
end

end

