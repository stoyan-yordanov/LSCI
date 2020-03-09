function lsci_gnrLaserSpeckle3D(SimulationType, ImageWidth, ImageHeight, Frames, ExposureTime, OutputFileType)
% Generate or simulate 3D Laser Speckle Image stack
% SimulationType = 'random-spceckle'
% ImageWidth = image width in px
% ImageHeight = image hieght in px
% Frames = number of frames to generate
% OutputFileType = 'tiff' (saves as multipage tiff file) | 'avi' (saves as grayscale uncompressed avi movie) | 'mj2' (Motion JPEG 2000 avi)
% ExposureTime = exposure time in [sec]

fprintf('\nStart Generating Laser Speckle... \n'); % show progress
startTime1 = tic;

 % Get laser speckle intensity using 'random-speckle' method
 switch(SimulationType)
     case 'random-speckle'
         lsp3DImageStack = randomLaserSpeckleIntensitySimulation(ImageHeight, ImageWidth, Frames, ExposureTime);     
     otherwise
         fprintf('\n\nUnsupported simulation type --> Simulation Type = %s\n', SimulationType);
         error('Exit due to the error above!');
 end
 
% Save result from Laser Speckle intensity 3D stack as tiff or video
baseFileName = ['LSP=3D' '_sim=' SimulationType '_xyz=' num2str(ImageWidth) 'x' num2str(ImageHeight) 'x' num2str(Frames) '_t=' num2str(ExposureTime) 's'];

% Save 3D stack frames as images
type3DStackItNormalization = 'global';
lsci_SaveToFrames(lsp3DImageStack, baseFileName, OutputFileType, type3DStackItNormalization);

% Show progress and stats
elapsedTime1 = toc(startTime1);

fprintf('\n\nEnd of processing --> 3D Stack (XYZ) = %dx%dx%d, ExposureTime = %.3f [ms]\n', ImageWidth, ImageHeight, Frames, ExposureTime*1000); % show progress
fprintf('Processing time = %.3f [sec]\n\n', elapsedTime1);

end

function rtrnLspIntensity = randomLaserSpeckleIntensitySimulation(ImageHeight, ImageWidth, Frames, ExposureTime)
% Simulate Laser Speckle Intensity Pattern
% Algorithm is based on the paper: Oliver Thompson et al, "Tissue perfusion measurements: multiple-exposure laser speckle analysis generates laser Doppler–like spectra", DOI: https://doi.org/10.1117/1.3400721

 % Init laser speckle intensity matrix
rtrnLspIntensity = zeros(ImageHeight, ImageWidth, Frames);

% Init the laser speckle matrix
lspMatrixE = complex(zeros(ImageHeight, ImageWidth));

% Init sub-matrix, a square the size L = 2*L'
submHeight = floor(ImageHeight/2);
submWidth = floor(ImageWidth/2);
subMatrixE = complex(ones(submHeight, submWidth)); % init the laser speckle sub-matrix

% Generate matrix with random phases in the range [-pi, pi]
phaseRateConst = 10000; % defines how fast the phases in the laser speckle change
rng(101, 'simdTwister'); % init rnd generator
phi = rand(submHeight, submWidth).*(2*pi) - pi; % rnd angles between [pi, -pi]
subMatrixE = exp(-1i*phi); % assign complex pahses to the sub matrix

% Generate laser spckle intensity 3D stack
fprintf('\nProgress, generate 3D Stack: 000.0 [%%] | 00000.0 [sec]');
for iZ = 1:Frames
    startTime2 = tic;
    
    % Calc electric field of laser speckle from random phases
    lspMatrixE = setLSPMatrix(subMatrixE, lspMatrixE); % set sub matrix on top of the LSP matrix
    fft2LspMatrixE = fft2(lspMatrixE);
    
    % Calc intensity from the elctric field distribution of laser speckle
    frameIt = conj(fft2LspMatrixE).*fft2LspMatrixE;
    rtrnLspIntensity(1:end, 1:end, iZ) = frameIt;
    
    % Recalc the new pahases - adds normally distributed phases
    phi = phi + randn(submHeight, submWidth).*(ExposureTime*phaseRateConst); % add normal rnd phases scaled by the time exposure and phase rate factor
    subMatrixE = exp(-1i*phi); % assign complex pahses to the sub matrix
    
    % Show progress
    elapsedTime2 = toc(startTime2);
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'); % delete previous progress line
    fprintf('%05.1f [%%] | %07.1f [sec]', (iZ/Frames)*100, (Frames - iZ)*elapsedTime2);
end

end

function OuterMatrix = setLSPMatrix(SubMatrix, OuterMatrix)
% Set the sub matrix to the center of the outer matrix

lspRows = size(OuterMatrix, 1);
lspColumns = size(OuterMatrix, 2);

submRows = size(SubMatrix, 1);
submColumns = size(SubMatrix, 2);

shiftHeight = floor((lspRows - submRows)/2) + 1;
shiftWidth = floor((lspColumns - submColumns)/2) + 1;

OuterMatrix(shiftHeight:(shiftHeight + submRows - 1), shiftWidth:(shiftWidth + submColumns - 1)) = SubMatrix;

end
