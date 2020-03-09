function lsci_tool3DImageFilesApplyBlur(InputFileType, OutputFileType)
% Apply image operations to files with 3D image stacks
% InputFileType = file type extension of the processed files
% OutputFileType = 'tiff' (saves as multipage tiff file) | 'avi' (saves as grayscale uncompressed avi movie) | 'mj2' (saves as uncompressed grayscale 16 bits video)

% Case no file provided or the string is not a valid file --> Get dir file list and choose a file t process
if strcmp(InputFileType, '')    
    fileDirFilter = '*';       
else
    fileDirFilter = ['*.' InputFileType];
end

% Get files and their number
fileList = lsci_sysGetDirectoryFileList(fileDirFilter); % return the list of file in the current dir
fileList = lsci_sysChooseFilesFromFileList(fileList); % get the file(s) to be processed
lengthFileList = size(fileList, 1);

if lengthFileList < 1
    fprintf('\nNo files matching the input file type found!\n');
    error('Exit due to above error!');
end

% Start processing
fprintf('\nStart Applying Blur to 3D Image Files... \n'); % show progress
startTime1 = tic;

% Optical system settings
LaserWavelengthUm = 0.633; % in [um]
NA = 0.28; % NA of the objective or imaging lens

% PSF settings of the optical/imaging system
PsfOpticalSwitch = 'on'; % 'on' | 'off', control if we add PSF to the observed image
PsfType = 'iairy'; % 'egauss' (Gaussian El. Field PSF) | 'igauss' (Gaussian intensity PSF)  | 'eairy' (El. Field Airy Disk) | 'iairy' (Intenstiy Airy Disk)
PsfRadiusUm = 1.22*LaserWavelengthUm/(2*NA); % in [um], radius of the PSF in the object space
PixelSizeUm = 1.38; % in [um], cameara pixel size after demagnification in the object space
PsfAmplitude = 1.0; % the amplitude of the PSF function
%SpeckleSizeOnCamera1Um = 1.22*LaserWavelengthUm/NA; % laser speckle size (image speckle) = PSF diameter
%SpeckleSizeOnCamera2Um = d*LaserWavelengthUm/a; % laser speckle size (object speckle) - d = object-camera distance, a - focus spot diameter

% Blurring/noise parameters to emulate noise
ScatteringLayerSwitch = 'off'; % 'on' | 'off', control if we add scattering layer (between sample and camera)
PsfKernelRadiusPx = 2*ceil(PsfRadiusUm/PixelSizeUm); % radius of the PSF kernel in [px], it must be bigger than PSF radius in order to emulate thick scattering layer
PsfKernelType = 'gauss-kernel'; % 'gauss-kernel' (Gauss PSF) | 'gauss-kernel-rnd-noise' (Gauss PSF + rnd(0,1) noise) | 'rnd-noise-pattern' (noise entire image rnd(0, 1)) | 'regular-grid-pattern' (grid of 0 and 1)
PsfKernelAmplitude = 1.0; % the amplitude of the PSF function

StaticScatteringPatternSwitch = 'off'; % 'on' | 'off', control if we add static scattering pattern
PatternLevel = 0.5; % intensity of the pattern, between [0, Infinity] from the mean intensity

BackgroundNoiseSwitch = 'off'; % 'on' | 'off', control if we add background noise
BackgroundLevel = 0.1; % background level, between [0, 1] from the max intensity

GaussNoiseSwitch = 'off'; % 'on' | 'off', control if we add gauss noise
GaussianNoiseLevel = 0.1; % noise level, between [0, 1] from the mean intensity

% Loop through files
for iFile = 1:lengthFileList
    fprintf('\n\n-----------------------------------------');
    fprintf('\nBEGIN: file #%d...', iFile);
    
    startTime2 = tic;
    
    % Get current input file
    InputFile = fileList{iFile, 1}; % only one file (the first one) will be processed
    
    % Check file name
    [filePath, fileName, fileExtension] = fileparts(InputFile);
    
    % Read input raw frames
    inXYZFrames = lsci_ReaderFramesToMatrix(InputFile, -1, -1, 'double'); % XY images array (Z = frame index)
    [ImageHeight, ImageWidth, Frames] = size(inXYZFrames);
    
    % Init 3D image stack
    out3DImageStack = zeros(ImageHeight, ImageWidth, Frames);
            
    % Init rnd number generator
    rng(101, 'simdTwister');
    
    % Init core structures
    if strcmp(PsfOpticalSwitch, 'on') == true
        psfOpticalSystemXY = getPsfOpticalSystem(ImageHeight, ImageWidth, PsfAmplitude, PsfRadiusUm, PsfType);
    end
    
    if strcmp(ScatteringLayerSwitch, 'on') == true 
        psfScatteringLayerXY = getPsfScatteringLayer(ImageHeight, ImageWidth, PsfKernelAmplitude, PsfKernelRadiusPx, PsfKernelType); % init scattering layer PSF
    end
    
    % Loop through the frames in each file
    for iFrame = 1:Frames
        tmpInputFrame = inXYZFrames(:, :, iFrame);

        % Add PSF of the optical system to the image
        if strcmp(PsfOpticalSwitch, 'on') == true
            tmpInputFrame = addPsfToImage(tmpInputFrame, psfOpticalSystemXY, 'conv2'); % convolve with the PSF of the optical system  
        end

        % Add different types of amplitude or/and phase blur/noise sources
        if strcmp(ScatteringLayerSwitch, 'on') == true        
            tmpInputFrame = addPsfToImage(tmpInputFrame, psfScatteringLayerXY, 'conv2'); % convolve with kernel to emulate blur from scattering layer
        else
            % Generate diffraction image by FFT propagation of the input Electrical field distribution
            %lspChImage = diffractionPropagatedImage(lspChImage, 'intensity'); % returns intensity distribution from the el. field
        end

        if strcmp(StaticScatteringPatternSwitch, 'on') == true        
            tmpInputFrame = addStaticScatteringPatternToImage(tmpInputFrame, PatternLevel, 'rect-net'); % add static scattering pattern
        end

        if strcmp(BackgroundNoiseSwitch, 'on') == true
            tmpInputFrame = addBackgroundNoiseToImage(tmpInputFrame, BackgroundLevel);% add backround noise (% from max intenisty)     
        end

        if strcmp(GaussNoiseSwitch, 'on') == true        
            tmpInputFrame = addGaussianSNRNoiseToImage(tmpInputFrame, GaussianNoiseLevel);  % add Gaussian noise (% from the background)     
        end

        % Add image to the image stack   
        out3DImageStack(1:end, 1:end, iFrame) = tmpInputFrame;          
    end
       
    
    % Build file name
    strDelimiter = '_';
    fileNameStrCellArray = splitFileNameByDelimiter(fileName, strDelimiter); % split file name in strings by delimiter
        
    psfString = sprintf('psf=%s(r=%.2fum)(%s)', PsfType, PsfRadiusUm, PsfOpticalSwitch);
    [fileNameStrCellArray, isPatternFound] = replaceStrPartsInStrCellArray(fileNameStrCellArray, 'psf=', psfString);
    if ~isPatternFound
        fileNameStrCellArray{end+1} = psfString;
    end
    
    blurString = sprintf('blur=(scl=%s,ssp=%s,bg=%s,gn=%s)', ScatteringLayerSwitch, StaticScatteringPatternSwitch, BackgroundNoiseSwitch, GaussNoiseSwitch);
    [fileNameStrCellArray, isPatternFound] = replaceStrPartsInStrCellArray(fileNameStrCellArray, 'blur=', blurString);
    if ~isPatternFound
        fileNameStrCellArray{end+1} = blurString;
    end

    % Save result as 3D stack file
    baseFileName = buildStringFromCellArray(fileNameStrCellArray, strDelimiter);
    
    % Save data as image or video
    fprintf('\n');
    type3DStackItNormalization = 'global';
    lsci_SaveToFrames(out3DImageStack, baseFileName, OutputFileType, type3DStackItNormalization);
    
    % Show progress
    elapsedTime2 = toc(startTime2);
    fprintf('\nEND: file #%d', iFile);
    fprintf('\nFiles Left #%d | %.1f [%%] | %.1f [sec]', (lengthFileList - iFile), (iFile/lengthFileList)*100, (lengthFileList - iFile)*elapsedTime2);
end

% Show progress and stats
elapsedTime1 = toc(startTime1);
fprintf('\n\nProcessing time = %.3f [sec]\n\n', elapsedTime1);

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

function rtrnPSF = calc2DPSF(PsfType, PsfRadius, X, Y)
% Calcs Intensity PSF

switch(PsfType)
    case 'egauss'
        PsfRadius = PsfRadius/1.452; % scale gaussian radius to fit best the Airy Disk with the given PSF radius
        rtrnPSF = ePsf2DGauss(PsfRadius, X, Y);
    case 'eairy'
        rtrnPSF = ePsf2DAiry(PsfRadius, X, Y);
    case 'igauss'
        PsfRadius = PsfRadius/1.452; % scale gaussian radius to fit best the Airy Disk with the given PSF radius
        rtrnPSF = iPsf2DGauss(PsfRadius, X, Y);
    case 'iairy'
        rtrnPSF = iPsf2DAiry(PsfRadius, X, Y);    
    otherwise
        fprintf('\n\nUnsupported PSF type --> PSF Type = %s\n', PsfType);
        error('Exit due to the error above!');
end

end

function rtrnPSF = ePsf2DGauss(w0, x, y)
% Calcs Gaussian Electrical PSF in (x,y) point
% PSFStruct = contains info (parameters) that define the PSF and help to calc it
% w0 = radius of the PSF (at 1/e)

% Calc intenisty in XY point
rtrnPSF = exp(-(x.^2 + y.^2)./w0^2);

end

function rtrnPSF = iPsf2DGauss(w0, x, y)
% Calcs Gaussian Intensity PSF in (x,y) point
% PSFStruct = contains info (parameters) that define the PSF and help to calc it
% w0 = radius of the PSF (at 1/e^2)

% Calc intenisty in XY point
rtrnPSF = exp(-2*(x.^2 + y.^2)./w0^2);

end

function rtrnPSF = ePsf2DAiry(r0, x, y)
% Calcs Airy disk Electrical Field PSF in (x,y) point
% PSFStruct = contains info (parameters) that define the PSF and help to calc it
% r0 = radius of the Airy disk (first zero of the Bessel function first kind first order J1)

% Calc the intensity of the Airy disk in XY point
r = (2*pi*0.61/r0).*sqrt(x.^2 + y.^2); % calc the argument of the Bessel function (first kind, first order)

if r > 0
    rtrnPSF = 2*besselj(1, r)./r; % calcs the Airy disk normalized to one
else
    % Calc in the limit r --> 0
    rtrnPSF = 1;
end

end

function rtrnPSF = iPsf2DAiry(r0, x, y)
% Calcs Airy disk Intensity PSF in (x,y) point
% PSFStruct = contains info (parameters) that define the PSF and help to calc it
% r0 = radius of the Airy disk (first zero of the Bessel function first kind first order J1)

% Calc the intensity of the Airy disk in XY point
r = (2*pi*0.61/r0).*sqrt(x.^2 + y.^2); % calc the argument of the Bessel function (first kind, first order)

if r > 0
    rtrnPSF = (2*besselj(1, r)./r).^2; % calcs the Airy disk normalized to one
else
    % Calc in the limit r --> 0
    rtrnPSF = 1;
end

end

function ImageXY = addBackgroundNoiseToImage(ImageXY, BackgroundLevel)
% Add background intensity to the 2D image (background is percentage from the maximum)

maxIt = max(ImageXY, [], 'all'); % get max intensity in the image
bgrIt = BackgroundLevel * maxIt; % calc background intensity constant, it is between [0, 1]*maxIt

ImageXY = ImageXY + bgrIt; % add background intensity to each pixel

end

function ImageXY = addGaussianSNRNoiseToImage(ImageXY, GaussianNoiseLevel)
% Add Gaussian noise to the 2D image (noise level is percentage from the mean)

[lengthX, lengthY] = size(ImageXY);

meanIt = sum(ImageXY, 'all')/(lengthX*lengthY); % get mean intensity in the image
noiseIt = GaussianNoiseLevel * meanIt; % calc noise intensity constant, it is between [0, 1]*meanIt

ImageXY = abs(ImageXY + noiseIt * randn(lengthX, lengthY)); % add noise intensity to each pixel

end

function ImageXY = addPsfToImage(ImageXY, PsfXY, TypeConvolution)
% Convolve with kernel to emulate PSF

switch(TypeConvolution)
    case 'conv2' % uses matlabs conv2() routine
        
        % Calc convolution of KernelPSF*ImageXY by conv2()
        ImageXY = conv2(PsfXY, ImageXY, 'same');
        
     case 'fft2' % uses FFT to calc convolution
        
        % Get dimenssions
        [imgRows, imgCols] = size(ImageXY);
        halfImgRows = round(imgRows/2);
        halfImgCols = round(imgCols/2);
        
        [psfRows, psfCols] = size(PsfXY);
        halfPsfRows = round(psfRows/2);
        halfPsfCols = round(psfCols/2);
        
        % Calc convolution of KernelPSF*ImageXY by FFT
        PsfXY = padarray(PsfXY, [halfImgRows, halfImgCols], 0, 'both'); % conv. needs zero padding
        ImageXY = padarray(ImageXY, [halfPsfRows, halfPsfCols], 0, 'both'); % conv. needs zero padding
        ImageXY = ifftshift(ifft2(fft2(PsfXY) .* fft2(ImageXY)));  % convolution with fft needs zero padding by leght1 + length2 - 1
        
        % Remove the zero padding
        ImageXY = ImageXY(halfImgRows:(halfImgRows + imgRows - 1), halfImgCols:(halfImgCols + imgCols - 1)); 
        
    otherwise
        fprintf('\n\nUnsupported convolution type --> Convolution Type = %s\n', KernelType);
        error('Exit due to the error above!');
end

end

function rtrnPsfOpticalSystemXY = getPsfOpticalSystem(ImageHeightPx, ImageWidthPx, PsfAmplitude, PsfRadiusPx, PsfType)
% Returns XY image representation of the PSF of the optical system

rows = ImageHeightPx;
cols = ImageWidthPx;
pixRowCenter = round(rows/2); % center of the image along the height
pixColCenter = round(cols/2); % center of the image alog the width

% Set the diameter of the PSF in terms of pixels
PsfDiameterPx = 2*PsfRadiusPx;

% Calc pixels window size - the area where we will calculate the kernel PSF function
psfWindow = struct();
psfWindow.minRow = floor(pixRowCenter - PsfDiameterPx);
psfWindow.minCol = floor(pixColCenter - PsfDiameterPx);
psfWindow.maxRow = ceil(pixRowCenter + PsfDiameterPx);
psfWindow.maxCol = ceil(pixColCenter + PsfDiameterPx);

% Check validity of the range
% if 2*PsfDiameterPx > rows || 2*PsfDiameterPx > cols
%     % Case the PSF window is too big
%     fprintf('\n\nPSF Kernel Diameter is bigger than image dimension(s) --> Decrease the size of PsfKernelRadiusPx\n');
%     error('Exit due to the error above!');
% end

% Init PSF kernel image
rtrnPsfOpticalSystemXY = zeros(rows, cols);

% Calc the kernel PSF image (it will be in the center of the image)
for iPixRow = psfWindow.minRow:psfWindow.maxRow
    if iPixRow > 0 && iPixRow <= rows % take into account only indexes that are within the image dimensions
        
        for iPixCol = psfWindow.minCol:psfWindow.maxCol
            
            if iPixCol > 0 && iPixCol <= cols % take into account only indexes that are within the image dimensions                
                rtrnPsfOpticalSystemXY(iPixRow, iPixCol) = PsfAmplitude * calc2DPSF(PsfType, PsfRadiusPx, pixRowCenter - iPixRow, pixColCenter - iPixCol);
            end
        end
    end
end        

end

function rtrnPsfScatteringLayerXY = getPsfScatteringLayer(ImageHeight, ImageWidth, PsfKernelAmplitude, PsfKernelRadiusPx, KernelType)
% Returns XY image representation of the PSF of the scattering layer

rows = ImageHeight;
cols = ImageWidth;
pixRowCenter = round(rows/2); % center of the image along the height
pixColCenter = round(cols/2); % center of the image alog the width

% Set the diameter of the PSF in terms of pixels
PsfDiameterPx = 2*PsfKernelRadiusPx;

% Calc pixels window size - the area where we will calculate the kernel PSF function
psfWindow = struct();
psfWindow.minRow = ceil(pixRowCenter - PsfDiameterPx);
psfWindow.minCol = ceil(pixColCenter - PsfDiameterPx);
psfWindow.maxRow = ceil(pixRowCenter + PsfDiameterPx);
psfWindow.maxCol = ceil(pixColCenter + PsfDiameterPx);

% Check validity of the range
% if 2*PsfDiameterPx > rows || 2*PsfDiameterPx > cols
%     % Case the PSF window is too big
%     fprintf('\n\nPSF Kernel Diameter is bigger than image dimension(s) --> Decrease the size of PsfKernelRadiusPx\n');
%     error('Exit due to the error above!');
% end

% Init PSF kernel image
rtrnPsfScatteringLayerXY = zeros(rows, cols);

switch(KernelType)
    case 'gauss-kernel' % pure gaussian kernel PSF emulating loss of resolution
        
        % Calc the kernel PSF image (it will be in the center of the image)
        for iPixRow = psfWindow.minRow:psfWindow.maxRow
            if iPixRow > 0 && iPixRow <= rows % take into account only indexes that are within the image dimensions
                
                for iPixCol = psfWindow.minCol:psfWindow.maxCol
                    
                    if iPixCol > 0 && iPixCol <= cols % take into account only indexes that are within the image dimensions
                        rtrnPsfScatteringLayerXY(iPixRow, iPixCol) = PsfKernelAmplitude * iPsf2DGauss(PsfKernelRadiusPx, pixRowCenter - iPixRow, pixColCenter - iPixCol);
                    end
                end
            end
        end        
        
     case 'gauss-kernel-rnd-noise' % gaussian kernel PSF + scattering noise
        
        % Calc the kernel PSF image (it will be in the center of the image)
        for iPixRow = psfWindow.minRow:psfWindow.maxRow
            if iPixRow > 0 && iPixRow <= rows % take into account only indexes that are within the image dimensions
                
                for iPixCol = psfWindow.minCol:psfWindow.maxCol
                    
                    if iPixCol > 0 && iPixCol <= cols % take into account only indexes that are within the image dimensions
                        rtrnPsfScatteringLayerXY(iPixRow, iPixCol) = PsfKernelAmplitude * iPsf2DGauss(PsfKernelRadiusPx, pixRowCenter - iPixRow, pixColCenter - iPixCol); % add gaussian kernel value
                        rtrnPsfScatteringLayerXY(iPixRow, iPixCol) = rtrnPsfScatteringLayerXY(iPixRow, iPixCol) * rand(1); % emulate scattering by rnd number in (0, 1) interval
                    end
                end
            end
        end        
                
    case 'rnd-noise-pattern' % scattering phase PSF pattern throught the entire image
        
        % Calc the kernel PSF image (it will be spread throughtout the entire image)
        %rtrnPsfScatteringLayerXY = rand(rows, cols); % emulate scattering by rnd scatters --> rnd number in (0, 1) interval
        
        % Calc the kernel PSF image (it will be in the center of the image)
        for iPixRow = psfWindow.minRow:psfWindow.maxRow
            if iPixRow > 0 && iPixRow <= rows % take into account only indexes that are within the image dimensions
                for iPixCol = psfWindow.minCol:psfWindow.maxCol
                    
                    if iPixCol > 0 && iPixCol <= cols % take into account only indexes that are within the image dimensions
                        rtrnPsfScatteringLayerXY(iPixRow, iPixCol) = PsfKernelAmplitude * round(rand(1)); % emulate scattering by rnd number in 0 or 1
                    end
                end
            end
        end
        
    case 'regular-grid-pattern' % scattering phase PSF pattern throught the entire image
        
        % Calc the kernel PSF image (it will be spread throughtout the entire image)
        %rtrnPsfScatteringLayerXY = rand(rows, cols); % emulate scattering by rnd scatters --> rnd number in (0, 1) interval
        
        % Calc the kernel PSF image (it will be in the center of the image)        
        freq = 1;
        for iPixRow = psfWindow.minRow:psfWindow.maxRow
            if iPixRow > 0 && iPixRow <= rows % take into account only indexes that are within the image dimensions
                
                for iPixCol = psfWindow.minCol:psfWindow.maxCol
                    
                    if iPixCol > 0 && iPixCol <= cols % take into account only indexes that are within the image dimensions
                        tmp = sin(2*pi*freq*(pixRowCenter - iPixRow)/(psfWindow.maxRow - psfWindow.minRow)) + cos(2*pi*freq*(pixColCenter - iPixCol)/(psfWindow.maxCol - psfWindow.minCol));
                        rtrnPsfScatteringLayerXY(iPixRow, iPixCol) = abs(tmp) * PsfKernelAmplitude; % emulate scattering by regular grid in o or 1
                    end
                end
            end
        end
        
    otherwise
        fprintf('\n\nUnsupported kernel type --> Kernel Type = %s\n', KernelType);
        error('Exit due to the error above!');
end

end

function ImageXY = addStaticScatteringPatternToImage(ImageXY, PatternLevel, PatternType)
% Add static scattering pattern

[rows, cols] = size(ImageXY);

meanIt = sum(ImageXY, 'all')/(rows*cols); % get mean intensity in the image
patternIt = PatternLevel * meanIt; % calc pattern intensity constant

% Init pattern image
patternImageXY = zeros(rows, cols);

switch(PatternType)
    case 'rect-net'
        % Add rectangular grid to the image
        numRowLines = 5;
        numColLines = 10;
        gapRows = floor(rows/numRowLines); % gap between the lines along height of the image
        gapCols = floor(cols/numColLines); % gap between the lines along width of the image
        lineTicknessPx = 2; % line thickness in [px]
        
        % Generate the pattern
        for iLine = 1:numRowLines % generate horizontal lines
            iRow = gapRows*(iLine - 1) + 1;
            patternImageXY(iRow:(iRow + lineTicknessPx - 1), 1:end) = patternIt;
        end
        
        for iLine = 1:numColLines % generate vertical lines
            iCol = gapCols*(iLine - 1) + 1;
            patternImageXY(1:end, iCol:(iCol + lineTicknessPx - 1)) = patternIt;
        end
        
        % Add the pattern to the image
        ImageXY = ImageXY + patternImageXY;
        
    otherwise
        fprintf('\n\nUnsupported pattern type --> Pattern Type = %s\n', PatternType);
        error('Exit due to the error above!');
end

end
