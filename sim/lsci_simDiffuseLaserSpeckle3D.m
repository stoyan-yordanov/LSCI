function lsci_simDiffuseLaserSpeckle3D(ImageWidth, ImageHeight, Frames, ExposureTime, OutputFileType)
% Simulate 3D Laser Speckle Image Stack from diffusion or/and flow (particles are assumed point like objects)
% ImageWidth = image width in px
% ImageHeight = image hieght in px
% Frames = number of frames to generate
% ExposureTime = exposure time in [sec]
% OutputFileType = 'tiff' (saves as multipage tiff file) | 'avi' (saves as grayscale uncompressed avi movie) | 'mj2' (saves as uncompressed grayscale 16 bits video)

%profile on;

% Set default number of threads
%lastNumThreads = maxNumCompThreads('automatic');
%lastNumThreads = maxNumCompThreads(8);

fprintf('\nStart Simulating Laser Speckle... \n'); % show progress
startTime1 = tic;

% BEGIN Additional Simulation Parameters
% BEGIN Additional Simulation Parameters

% Image snapshot settings
FrameSnapshotSwitch = 'on'; % 'on' (save snapshot of the current frame) | 'off' (disable snapshot saving)
FrameSnapshotWriteMode = 'overwrite'; % 'append' (append to existing image) | 'overwrite' (overwrite existing image)
FrameSnapshotBaseFileName = 'tmp-frame-snapshot';

% Init global variables
global glLSP;
glLSP = struct(); % holds global variables used throughout the function
glLSP.RSCoreLookUpTbl = []; % lookup table for the Rayleigh-Sommerfeld diffraction wavelet
glLSP.TypeWavelet = 'rsw'; % 'plw' (Plane Wave propagator) | 'rsw' (Rayleigh-Sommerfeld Wave propagator)
glLSP.RandomizeUPhaseSwitch = 'dif'; % 'on' | 'off' randomize the phase of each Ufield wave to emulate multiple scattering
glLSP.StepSizeUmLookUpTbl = 0.1; % step size in [um] along XYZ for the desired precison of the source-point distance calculations
glLSP.ObservationPlaneXYZUm = [];% gets the observation planes pixels XY and Z coordinates in [um]
glLSP.ParticlesSnapshotSwitch = 'off'; % 'on' (save particles snapshot) | 'off' (do not save particles snapshot)
glLSP.particlesSnapshot = {}; % cell array to store snapshots of particles' positions

% Multiexposure settings --> note: enabling multiexposure increases time step up to the 1/Fps duaration, after which it starts again
% If multiexposure is enabled then every subsequent frame increases exposure time depending on the multi exposure algoritm
MultiExposureAlgorithm = 'off'; % frame/multiexposure algorithm - 'off' (disable, i.e. dt = exposure time) | 'lin' (new dt = dt + exposure time) | 'pow2' (new dt = 2*dt)

% Camera and time settings
FpsSwitch = 'off'; % 'on' (enabled) or 'off' (disable) frame per second limit (if true => 1/Fps - ExposureTime wating time bewteen frames)
Fps = 1000; % frames per second (used only if FpsSwitch = 'on', otherwise Fps = 1/ExposureTime)
TimeStep = ExposureTime; % time step in [s]
%IntensityConst = 1000; % intensity per time constant (e.g. Intensity = IntensityConst*TimeStep)

% Optical system settings
Magnification = 5.01;
LaserWavelengthUm = 0.633; % in [um]
RefIndexImmersion = 1.0; % ref index of the optical system immersion liquid
RefIndexMedium = 1.33; % ref index of the medium where the particles diffuse
NA = 0.28; % NA of the objective or imaging lens
CameraPixelSizeUm = 6.9; % in [um], cameara physical pixel size

% PSF settings of the optical/imaging system
PsfOpticalSwitch = 'on'; % 'on' | 'off', control if we add PSF to the observed image
PsfType = 'iairy'; % 'egauss' (Gaussian El. Field PSF) | 'igauss' (Gaussian intensity PSF)  | 'eairy' (El. Field Airy Disk) | 'iairy' (Intenstiy Airy Disk)
PsfRadiusUm = 1.22*LaserWavelengthUm/(2*NA); % in [um], radius of the PSF in the object space
PixelSizeUm = CameraPixelSizeUm/Magnification; % in [um], cameara pixel size after demagnification in the object space
PsfAmplitude = 1.0; % the amplitude of the PSF function
%SpeckleSizeOnCamera1Um = 1.22*LaserWavelengthUm/NA; % laser speckle size (image speckle) = PSF diameter
%SpeckleSizeOnCamera2Um = d*LaserWavelengthUm/a; % laser speckle size (object speckle) - d = object-camera distance, a - focus spot diameter

NumberParticles = 10000; % number of particles to propagate
NumberStaticScatters = 20000; % number of statically scattering particles
ChannelHeightRatio = 16; % image/channel height ratio
ChannelWidtRatio = 1; % image/channel width ratio

ObservationPlaneZUm = -1; % observation plane Z coordinate in [um] - in there we calc the resulting intensity/electrical field
WallOffset.x = 0; % not used since the flow direction is along X
WallOffset.y = 0; % the offset of the wall of the channel along Y
WallOffset.z = 0; % the offset of the wall of the channel along Z

DiffusionCoefficient.Dx = 0.0; % in [um^2/s] --> 5 [um^2/s] is ca. Rh = 50 nm
DiffusionCoefficient.Dy = 0.0; % in [um^2/s] --> 5 [um^2/s] is ca. Rh = 50 nm
DiffusionCoefficient.Dz = 0.0; % in [um^2/s] --> 5 [um^2/s] is ca. Rh = 50 nm
FlowVelocityUm.vx = 1000.0; % in [um/s], flow velocity along X
FlowVelocityUm.vy = 0.0; % in [um/s], , flow velocity along Y (not implemented)
FlowVelocityUm.vz = 0.0; % in [um/s], , flow velocity along Z (not implemented)

DiffusionStaticScatters.Dx = 0.0; % in [um^2/s] --> 5 [um^2/s] is ca. Rh = 50 nm
DiffusionStaticScatters.Dy = 0.0; % in [um^2/s] --> 5 [um^2/s] is ca. Rh = 50 nm
DiffusionStaticScatters.Dz = 0.0; % in [um^2/s] --> 5 [um^2/s] is ca. Rh = 50 nm

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

% END Additional Simulation Parameters
% END Additional Simulation Parameters

% Init laser speckle intensity matrix
lsp3DImageStack = zeros(ImageHeight, ImageWidth, Frames);

% Init the laser speckle matrix
imageHeightUm = ImageHeight*PixelSizeUm;
imageWidthUm = ImageWidth*PixelSizeUm;
lspMatrix = zeros(ImageHeight, ImageWidth);

% Micro channel image matrix --> a represent a channel inside the bigger matrix
chHeightPx = floor(ImageHeight/ChannelHeightRatio);
chWidthPx = floor(ImageWidth/ChannelWidtRatio);
chDepthPx = 20;
chDepthSSPx = 5;
chHeightUm = chHeightPx*PixelSizeUm; % in [um]
chWidthUm = chWidthPx*PixelSizeUm; % in [um]
chDepthUm = chDepthPx*PixelSizeUm; % in [um]
chDepthSSUm = chDepthSSPx*PixelSizeUm; % in [um]

% Init rnd number generator
rng(101, 'simdTwister');

% Init core simulation structures
glLSP.ObservationPlaneXYZUm = getPixelsCenterXY(chHeightPx, chWidthPx, PixelSizeUm); % gets the observation plane pixels XY coordinates in [um]
glLSP.ObservationPlaneXYZUm.Z = ObservationPlaneZUm; % observation plane Z coordinate in [um] - in there we calc the resulting intensity/electrical field
glLSP.RSCoreLookUpTbl = setRSCoreLookUpTbl(glLSP.TypeWavelet, glLSP.StepSizeUmLookUpTbl, glLSP.ObservationPlaneXYZUm, chHeightUm, chWidthUm, chDepthUm, RefIndexMedium, LaserWavelengthUm);

psfOpticalSystemXY = getPsfOpticalSystem(chHeightPx, chWidthPx, PsfAmplitude, PsfRadiusUm, PsfType);
psfScatteringLayerXY = getPsfScatteringLayer(chHeightPx, chWidthPx, PsfKernelAmplitude, PsfKernelRadiusPx, PsfKernelType); % init scattering layer PSF
particles = generateParticles(NumberParticles, WallOffset, chHeightUm, chWidthUm, chDepthUm, DiffusionCoefficient); % init and generate particles and their initial XYZ coordinates, diffusion etc
staticScatters = generateParticles(NumberStaticScatters, WallOffset, chHeightUm, chWidthUm, chDepthSSUm, DiffusionStaticScatters); % init and generate static scatters and their initial XYZ coordinates, diffusion etc

% Generate laser spckle intensity 3D stack
fprintf('\nProgress, generate 3D Stack: 000.0 [%%] | 00000.0 [sec]');
for iZ = 1:Frames
    startTime2 = tic;
    
    % Propagate particles in the channel by the given time step
    particles = updateParticlesPosition(particles, TimeStep, WallOffset, chHeightUm, chWidthUm, chDepthUm, FlowVelocityUm, glLSP.RandomizeUPhaseSwitch);
    
    % Particles location snapshot of first and last positions
    if strcmp(glLSP.ParticlesSnapshotSwitch, 'on')
        getParticlesSnapshot(particles, iZ, [1, Frames]);
    end
        
    % Generate Intensity distribution image from particle coordinates (Blur the LSP Image using convolution with the PSF)    
    lspChImage = lsci_simGenerateDiffusiveIntensityImage_mex(particles, staticScatters, uint32(chHeightPx), uint32(chWidthPx), LaserWavelengthUm, glLSP.RSCoreLookUpTbl, glLSP.ObservationPlaneXYZUm, glLSP.StepSizeUmLookUpTbl); % the diffraction pattern in the XY plane
    %lspChImage = generateIntensityImage(particles, staticScatters, chHeightPx, chWidthPx, LaserWavelengthUm, glLSP.RSCoreLookUpTbl, glLSP.ObservationPlaneXYZUm, glLSP.StepSizeUmLookUpTbl); % the diffraction pattern in the XY plane
    
    % Generate diffraction intensity image from the input Electrical field distribution
    %lspChImage = generateEFieldImage(particles, chHeightPx, chWidthPx, RefIndexMedium, LaserWavelengthUm, PixelSizeUm);
    %lspChImage = calcEfieldToIntensityImage(lspChImage, 'intensity'); % returns intensity distribution of the resulting el. field
    
    % Add PSF of the optical system to the image
    if strcmp(PsfOpticalSwitch, 'on') == true
        lspChImage = addPsfToImage(lspChImage, psfOpticalSystemXY, 'conv2'); % convolve with the PSF of the optical system  
    end
    
    % Add different types of amplitude or/and phase blur/noise sources
    if strcmp(ScatteringLayerSwitch, 'on') == true        
        lspChImage = addPsfToImage(lspChImage, psfScatteringLayerXY, 'conv2'); % convolve with kernel to emulate blur from scattering layer
    else
        % Generate diffraction image by FFT propagation of the input Electrical field distribution
        %lspChImage = diffractionPropagatedImage(lspChImage, 'intensity'); % returns intensity distribution from the el. field
    end
    
    if strcmp(StaticScatteringPatternSwitch, 'on') == true        
        lspChImage = addStaticScatteringPatternToImage(lspChImage, PatternLevel, 'rect-net'); % add static scattering pattern
    end
    
    if strcmp(BackgroundNoiseSwitch, 'on') == true
        lspChImage = addBackgroundNoiseToImage(lspChImage, BackgroundLevel);% add backround noise (% from max intenisty)     
    end
    
    if strcmp(GaussNoiseSwitch, 'on') == true        
        lspChImage = addGaussianSNRNoiseToImage(lspChImage, GaussianNoiseLevel);  % add Gaussian noise (% from the background)     
    end
        
    % Merge laser speckle sub image to the whole image    
    frameIt = setLSPMatrix(lspChImage, lspMatrix); % set sub image on top of the LSP matrix
    lsp3DImageStack(1:end, 1:end, iZ) = frameIt;
    %lsp3DImageStack(1:end, 1:end, iZ) = IntensityConst*TimeStep*frameIt; % scale intensity poprotional to the time step duaration
    
    % Save snapshot of the rendered image(s)
    if strcmp(FrameSnapshotSwitch, 'on')
        lsci_SaveSingleFrame(frameIt, FrameSnapshotBaseFileName, FrameSnapshotWriteMode);
    end
    
    % Propagate particles in dark mode (i.e. no contrib. to the image) --> emulates that camera frame rate is slower than camera exposure time duaration
    if strcmp(FpsSwitch, 'on') == true
        % Make step after each frame to emulate the frame transfer time and thus the limits due to Fps (Frames per second)
        darkTimeStep = 1/Fps - TimeStep; % dark (no intensity added to image) time step
        
        % Check that dark time step is no negative
        if darkTimeStep < 0
            fprintf('\n\n Exposure time is bigger than Frame Rate duaration --> Exposure = %f, Fps Time = %f\n', ExposureTime, 1/Fps);
            error('Exit due to the error above!');
        end
        
        % Update particle in dark mode (i.e. without contributing to the final image)
        particles = updateParticlesPosition(particles, darkTimeStep, WallOffset, chHeightUm, chWidthUm, chDepthUm, FlowVelocityUm, glLSP.RandomizeUPhaseSwitch);
    else        
        Fps = 1/ExposureTime; % frame rate given by the exposure time duaration        
    end
    
    % Multiexposure update - case multiexposure is not 'off' => set new time step according to chosen algorithm
    switch(MultiExposureAlgorithm)
        case 'off'
            % Do nothing - time step stays the same
            %TimeStep = TimeStep;
        case 'lin'
            TimeStep = TimeStep + ExposureTime; % changes in a linear fashion by a step = exposure time            
        case 'pow2'
            TimeStep = 2*TimeStep; % change as power of 2 of the exposure time
        otherwise
            fprintf('\n\nYou have chosen unsupported multiexposure algorithm --> Multiexposure Algorithm = %s\n', MultiExposureAlgorithm);
            error('Exit due to the error above!');
    end
        
    % Set time step limits in case Fps option is 'on' (then time step will be reset when exceeding the frame time duaration)
    if strcmp(FpsSwitch, 'on') == true
        FrameTime = 1/Fps; % time between frames
        
        %Check if new time step does not exceed time between frames
        if TimeStep > FrameTime
            TimeStep = ExposureTime; % reset time step to the initial exposure
        end
    end
            
    % Show progress
    elapsedTime2 = toc(startTime2);
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'); % delete previous progress line
    fprintf('%05.1f [%%] | %07.1f [sec]', (iZ/Frames)*100, (Frames - iZ)*elapsedTime2);    
end

% Save result from Laser Speckle intensity 3D stack as tiff or video
strBlur = '_blur=(';
if strcmp(ScatteringLayerSwitch, 'on')
    strBlur = [strBlur 'scl,'];    
end

if strcmp(StaticScatteringPatternSwitch, 'on')
    strBlur = [strBlur 'ssp,'];
end

if strcmp(BackgroundNoiseSwitch, 'on')
    strBlur = [strBlur 'bg,'];
end

if strcmp(GaussNoiseSwitch, 'on')
    strBlur = [strBlur 'gn'];
end

if strcmp(strBlur, '_blur=(')
    % Case no active switch --> set blur string to empty
    strBlur = '';
else
    if strcmp(strBlur(end), ',')
        strBlur = strBlur(1:end-1); % remove comma at the end of the blur string
    end
    
    strBlur = [strBlur ')'];
end

baseFileName = ['LSP=3D' sprintf('_wv=%s(Uph=%s)', glLSP.TypeWavelet, glLSP.RandomizeUPhaseSwitch) sprintf('_z=%.1fum', ObservationPlaneZUm) sprintf('_xytz=%dx%dx%dx%d', ImageWidth, ImageHeight, Frames, chDepthPx)...
    sprintf('_me=%s', MultiExposureAlgorithm) '_t=' num2str(ExposureTime) 's' sprintf('_fps=%d(%s)', int32(Fps), FpsSwitch)...
    sprintf('_px=%.2fum', PixelSizeUm) sprintf('_psf=%s(r=%.2fum)(%s)', PsfType, PsfRadiusUm, PsfOpticalSwitch)...
    sprintf('_np=%d', int32(NumberParticles)) sprintf('_Dxyz=(%.1f,%.1f,%.1f)um2s', DiffusionCoefficient.Dx, DiffusionCoefficient.Dy, DiffusionCoefficient.Dz) sprintf('_Vxy=(%.0f,%.0f)ums', FlowVelocityUm.vx, FlowVelocityUm.vy)...
    sprintf('%s', strBlur)];

% Save data as image or video
fprintf('\n');
type3DStackItNormalization = 'global';
lsci_SaveToFrames(lsp3DImageStack, baseFileName, OutputFileType, type3DStackItNormalization);

% Save particles snapshots as 3D image frames
if strcmp(glLSP.ParticlesSnapshotSwitch, 'on')
    saveParticlesSnapShot(baseFileName, PixelSizeUm, chHeightPx, chWidthPx, chDepthPx);
end

% Show progress and stats
elapsedTime1 = toc(startTime1);

fprintf('\nEnd of processing --> 3D Stack (XYZ) = %dx%dx%d, ExposureTime = %.3f [ms]\n', ImageWidth, ImageHeight, Frames, ExposureTime*1000); % show progress
fprintf('Processing time = %.3f [sec]\n\n', elapsedTime1);

%profile viewer;

% Set default number of threads
%lastNumThreads = maxNumCompThreads('automatic');

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

function rtrnParticles = generateParticles(NumberParticles, WallOffset, ChHeightUm, ChWidthUm, ChDepthUm, DiffusionCoefficient)
% Inititalize particles and their corrdinates, diffusion etc

rtrnParticles = struct([]);

% Generate the vectors of the initial position to be within the channel limits
posUm.X = WallOffset.x + rand(NumberParticles, 1).*(ChWidthUm - WallOffset.x); % X vector of random initial positions in [um]
posUm.Y = WallOffset.y + rand(NumberParticles, 1).*(ChHeightUm - WallOffset.y); % Y vector of random initial positions in [um]
posUm.Z = WallOffset.z + rand(NumberParticles, 1).*(ChDepthUm - WallOffset.z); % Z vector of random initial positions in [um]

% Assign initial particle position etc
for i = 1:NumberParticles
    % Generate XY positions --> in the begining old and new positions are the same
    rtrnParticles(i).x0 = posUm.X(i); % store old X position
    rtrnParticles(i).y0 = posUm.Y(i); % store old Y position
    rtrnParticles(i).z0 = posUm.Z(i); % store old Z position
    
    rtrnParticles(i).x = posUm.X(i); % new X position
    rtrnParticles(i).y = posUm.Y(i); % new Y position
    rtrnParticles(i).z = posUm.Z(i); % new Z position
    rtrnParticles(i).Dx = DiffusionCoefficient.Dx;
    rtrnParticles(i).Dy = DiffusionCoefficient.Dy;
    rtrnParticles(i).Dz = DiffusionCoefficient.Dz;
    rtrnParticles(i).Ufield = 1.0; % the init amplitude and phase of the electric field for this particle
end

end

function Particles = updateParticlesPosition(Particles, TimeStep, WallOffset, ChHeightUm, ChWidthUm, ChDepthUm, FlowVelocityUm, RandomizeUPhaseSwitch)
% Update particles corrdinates, diffusion etc

numberParticles = length(Particles);

% Generate the normal rnd numbers for the vectors of the position changes
rndNormal = struct();
rndNormal.x = randn(numberParticles, 1);
rndNormal.y = randn(numberParticles, 1);
rndNormal.z = randn(numberParticles, 1);

% Generate the discrete rnd numbers (-1,+1) mathcinf mean and variance ot normal rnd numbers for the vectors of the position changes
% rndNormal = struct();
% rndNormal.x = 2*round(rand(numberParticles, 1)) - 1;
% rndNormal.y = 2*round(rand(numberParticles, 1)) - 1;
% rndNormal.z = 2*round(rand(numberParticles, 1)) - 1;

% Generate init amplitude and phase
%Uamp = rand(numberParticles, 1); % vector of random initial amplitudes (simulates multiple scattering) - direct generation (0, 1)
Uamp = 1.0; % vector of initial amplitude(s)
if strcmp(RandomizeUPhaseSwitch, 'on')
    Uphase = exp(1i .* 2 .* pi .* rand(numberParticles, 1)); % vector of random initial phases (simulates multiple scattering)
end

% Update particles position, update equation is diffusion + flow --> r(t + dt) = r(t) + v(r(t))*dt + sqrt(2*D*dt)*gaussNoise
% In our case flow velocity is constant through channel => r(t + dt) = r(t) + v*dt + sqrt(2*D*dt)*gaussNoise
for i = 1:numberParticles
    % Store current position before the update
    Particles(i).x0 = Particles(i).x;
    Particles(i).y0 = Particles(i).y;
    Particles(i).z0 = Particles(i).z;
    
    % Update current position --> origin of coordinate system is at X = 0, Y = 0, Z = 0, and the box is within the positive XYZ quadrant
    Particles(i).x = Particles(i).x0 + FlowVelocityUm.vx*TimeStep + sqrt(2*Particles(i).Dx*TimeStep)*rndNormal.x(i);
    Particles(i).y = Particles(i).y0 + FlowVelocityUm.vy*TimeStep + sqrt(2*Particles(i).Dy*TimeStep)*rndNormal.y(i); 
    Particles(i).z = Particles(i).z0 + FlowVelocityUm.vz*TimeStep + sqrt(2*Particles(i).Dz*TimeStep)*rndNormal.z(i); 
    
    % Update amplitude and phase of e-field
    if strcmp(RandomizeUPhaseSwitch, 'on')
        Particles(i).Ufield = Uamp .* Uphase(i); % the amplitude and phase of the electric field for this particle
    else
        Particles(i).Ufield = Uamp;
    end
    
    % Check for boundary collisions - the origin of coordinate system is at the top left corner of the window (x-axiss points to the right, y-axis points down)
    if Particles(i).x > ChWidthUm
        % Case particle wnet out of the right border of the window => generate new particle on the left border
        Particles(i).x = Particles(i).x - ChWidthUm; % particle goes on the left border
        Particles(i).y = rand(1)*ChHeightUm; % Y is within the channel border
        %Particles(i).z = rand(1)*ChDepthUm; % Y is within the channel border
        
        % Reset old position (since we generate new particle position)
        Particles(i).x0 = 0;
        Particles(i).y0 = Particles(i).y;
        %Particles(i).z0 = Particles(i).z;
    elseif Particles(i).x < 0
        % Case particle went out of the left border of the window => generate new particle on the right border
        Particles(i).x = ChWidthUm + Particles(i).x; % particle goes on the right border
        Particles(i).y = rand(1)*ChHeightUm; % Y is within the channel border
        %Particles(i).z = rand(1)*ChDepthUm; % Y is within the channel border
        
        % Reset old position (since we generate new particle position)
        Particles(i).x0 = ChWidthUm;
        Particles(i).y0 = Particles(i).y;
        %Particles(i).z0 = Particles(i).z;
    end
    
    % Double check y position
    if Particles(i).y < WallOffset.y % offset from the Y origin
        % Case particle went out of the top border of the window => make reflection
        Particles(i).y = WallOffset.y - (Particles(i).y - WallOffset.y); % reflection due to collision at the given Y  
    elseif Particles(i).y > ChHeightUm
        % Case particle went out of the bottom border of the window => make reflection
        Particles(i).y = ChHeightUm - (Particles(i).y - ChHeightUm); % Y is within the channel border
    end
    
    if Particles(i).y < WallOffset.y % offset from the Y origin
        % Case particle went out of the top border of the window => make reflection
        Particles(i).y = WallOffset.y - (Particles(i).y - WallOffset.y); % reflection due to collision at the given Y  
    elseif Particles(i).y > ChHeightUm
        % Case particle went out of the bottom border of the window => make reflection
        Particles(i).y = ChHeightUm - (Particles(i).y - ChHeightUm); % Y is within the channel border
    end
    
    % Double check z position
    if Particles(i).z < WallOffset.z % offset from the Z origin
        % Case particle went out of the top border of the window => make reflection
        Particles(i).z = WallOffset.z - (Particles(i).z - WallOffset.z); % reflection due to collision with the wall at given Z
    elseif Particles(i).z > ChDepthUm
        % Case particle went out of the bottom border of the window => make reflection
        Particles(i).z = ChDepthUm - (Particles(i).z - ChDepthUm); % Z is within the channel border
    end
    
    if Particles(i).z < WallOffset.z % offset from the Z origin
        % Case particle went out of the top border of the window => make reflection
        Particles(i).z = WallOffset.z - (Particles(i).z - WallOffset.z); % reflection due to collision with the wall at given Z
    elseif Particles(i).z > ChDepthUm
        % Case particle went out of the bottom border of the window => make reflection
        Particles(i).z = ChDepthUm - (Particles(i).z - ChDepthUm); % Z is within the channel border
    end
end

end

function rtrnIntensityImage = generateIntensityImage(Particles, StaticScatters, ImageHeightPx, ImageWidthPx, LaserWavelengthUm, RSCoreLookUpTbl, ObservationPlaneXYZUm, StepSizeXYZUm)
% Generate Intensity field diffraction pattern in XY plane from particle positions by E-field wave propagation

% global glLSP;
% RSCoreLookUpTbl = glLSP.RSCoreLookUpTbl;
% ObservationPlaneXYZUm = glLSP.ObservationPlaneXYZUm;
% StepSizeXYZUm = glLSP.StepSizeUmLookUpTbl;

numberParticles = length(Particles);
numberStaticScatters = length(StaticScatters);
[rsRows, rsCols] = size(RSCoreLookUpTbl);
rtrnIntensityImage = zeros(ImageHeightPx, ImageWidthPx);
%k = 2*pi*RefIndexMedium/LaserWavelengthUm; % wave number

% Calc particles displacement --> dx, dy, dz, dr
dx = zeros(numberParticles, 1);
dy = zeros(numberParticles, 1);
dz = zeros(numberParticles, 1);
dr = zeros(numberParticles, 1);
for i = 1:numberParticles % loop thorough the particles
    dx(i) = Particles(i).x - Particles(i).x0;
    dy(i) = Particles(i).y - Particles(i).y0;
    dz(i) = Particles(i).z - Particles(i).z0;
    dr(i) = sqrt(dx(i)^2 + dy(i)^2 + dz(i)^2); % length of the position change
end
maxdr = max(dr); % max dr displacement
%numSubSteps = floor(2*maxdr/LaserWavelengthUm + 1); % number of steps to walk between old and new position, each step moves a length equal half the wavelength
numSubSteps = floor(4*maxdr/LaserWavelengthUm + 1); % number of steps to walk between old and new position, each step moves a length equal 1/10 of the wavelength
dx = dx./numSubSteps; % fine dx step
dy = dy./numSubSteps; % fine dy step
dz = dz./numSubSteps; % fine dz step

% Calc intensity image
for iSubStep = 1:numSubSteps
    tmpUfieldImage = complex(zeros(ImageHeightPx, ImageWidthPx)); % reset
    tmpUfieldScattersImage = complex(zeros(ImageHeightPx, ImageWidthPx)); % reset
      
    % Calc E-field due to static scatters and secondary waves due to moving particles    
    parfor j = 1:numberStaticScatters % loop through the particles
        SSUfield = complex(StaticScatters(j).Ufield);
        
        % Calc the diffuse scattering E-field due to static scatters
        for i = 1:numberParticles
            
            % Fine propagation to calc Electrical field diffraction pattern        
            psRx = Particles(i).x0 - StaticScatters(j).x0; % particle-particle position
            psRy = Particles(i).y0 - StaticScatters(j).y0; % particle-particle position
            psRz = abs(Particles(i).z0 - StaticScatters(j).z0); % particle-particle position   
            
            % Calc XY distance particle-static_scatter in Matrix form                    
            psR = sqrt(psRx.^2 + psRy.^2 + psRz.^2); % max absolute distance
            
            if (psR < StepSizeXYZUm) % check that particles do not become closer than a min distance
                psR = StepSizeXYZUm;
            end
            
            % Calc indexes for the lookup table value in Matrix form
            isR = floor(psR./StepSizeXYZUm) + 1;
            isRz = floor(psRz./StepSizeXYZUm) + 1;
            
            % Add the contribution of particle i to the Ufield of particle j            
            SSUfield = SSUfield + 1.0 .* Particles(i).Ufield .* RSCoreLookUpTbl(isR, isRz); % light scattering coefficient            
        end
        
        % Calc Z distance particle-pixel
        Rz = StaticScatters(j).z0 - ObservationPlaneXYZUm.Z; % observation plane is along Z
        iRz = ceil(Rz./StepSizeXYZUm);
        
        % Calc XY distance particle-pixel in Matrix form
        Rx = StaticScatters(j).x0 - ObservationPlaneXYZUm.X;
        Ry = StaticScatters(j).y0 - ObservationPlaneXYZUm.Y;
        %Rz = abs(StaticScatters(j).z0 - ObservationPlaneXYZUm.Z);
        R = sqrt(Rx.^2 + Ry.^2 + Rz.^2); % max absolute distance source-point
        
        % Calc indexes for the lookup table value in Matrix form
        iR = ceil(R./StepSizeXYZUm);
        %iRz = ceil(Rz./StepSizeXYZUm);
        
        % % Assign intensity to the given pixel based on the particle position --> vectorized form
        tmpUfieldScattersImage = tmpUfieldScattersImage + SSUfield .* reshape(RSCoreLookUpTbl(iR(:), iRz), [ImageHeightPx, ImageWidthPx]); % precalc the RS core array for the given particle
    end
    
    % Calc E-field due to moving particles
    parfor i = 1:numberParticles % loop through the particles
        
        % Init U-field of particle i
       Ufield = Particles(i).Ufield;
        
        % Calc the diffuse scattering E-field on particle i (the sum of all U-fields due to the other particles)
%         for j = 1:numberParticles % loop through the particles
%             
%             % Exclude from calc the current particle i
%             if j == i
%                continue;
%             end
%             
%             % Fine propagation to calc Electrical field diffraction pattern        
%             ppRx = Particles(i).x0 - Particles(j).x0; % particle-particle position
%             ppRy = Particles(i).y0 - Particles(j).y0; % particle-particle position
%             ppRz = abs(Particles(i).z0 - Particles(j).z0); % particle-particle position   
%             
%             % Calc XY distance particle-particle in Matrix form                    
%             ppR = sqrt(ppRx.^2 + ppRy.^2 + ppRz.^2); % max absolute distance particle-particle
%             
%             if (ppR < StepSizeXYZUm) % check that particles do not become closer than a min distance
%                 ppR = StepSizeXYZUm;
%             end
%             
%             % Calc indexes for the lookup table value in Matrix form
%             ipR = floor(ppR./StepSizeXYZUm) + 1;
%             ipRz = floor(ppRz./StepSizeXYZUm) + 1;
% 
%             % Add the contribution of particle j to the Ufield of particle i
%             Ufield = Ufield + 1.0 .* Particles(j).Ufield .* RSCoreLookUpTbl(ipR, ipRz); % 10% light scattering coefficient
%             %Ufield = Ufield + 0.1 .* Particles(j).Ufield .* RSCoreLookUpTbl(ipR, ipRz); % 10% light scattering coefficient
%             %Ufield = Ufield + Particles(j).Ufield .* RSCoreLookUpTbl(ipR, ipRz); % 100% light scattering coefficient
%         end
        
        % Monte Carlo photon generation and propagation
        %propagatePhotons(Particle, PhotonDiffusion, NumberPhotons, ScatteringLengthUm, WallOffset, ChHeightUm, ChWidthUm, ChDepthUm)
        
        % Calc Z distance particle-pixel
        Rz = Particles(i).z0 - ObservationPlaneXYZUm.Z; % observation plane is along Z
        iRz = ceil(Rz./StepSizeXYZUm);
        
        % Calc XY distance particle-pixel in Matrix form
        Rx = Particles(i).x0 - ObservationPlaneXYZUm.X;
        Ry = Particles(i).y0 - ObservationPlaneXYZUm.Y;
        %Rz = Particles(i).z0 - ObservationPlaneXYZUm.Z;        
        R = sqrt(Rx.^2 + Ry.^2 + Rz.^2); % max absolute distance source-point
        
        % Calc indexes for the lookup table value in Matrix form
        iR = ceil(R./StepSizeXYZUm);
        %iRz = ceil(Rz./StepSizeXYZUm);
        
        % % Assign intensity to the given pixel based on the particle position --> vectorized form
        tmpUfieldImage = tmpUfieldImage + Ufield .* reshape(RSCoreLookUpTbl(iR(:), iRz), [ImageHeightPx, ImageWidthPx]); % precalc the RS core array for the given particle
    end
    
    % Sum E-field of moving particles and static scatters
    tmpUfieldImage = tmpUfieldImage + tmpUfieldScattersImage;
    
    % Add E-field image to the Intensity image
    rtrnIntensityImage = rtrnIntensityImage + conj(tmpUfieldImage) .* tmpUfieldImage;
    
    % Update particles positions
    for iP = 1:numberParticles % loop through the particles
        
        % Fine propagation to calc Electrical field diffraction pattern        
        Particles(iP).x0 = Particles(iP).x0 + dx(iP); % update position
        Particles(iP).y0 = Particles(iP).y0 + dy(iP); % update position
        Particles(iP).z0 = Particles(iP).z0 + dz(iP); % update position 
    end
    
end

% Average over the num of substeps
rtrnIntensityImage = rtrnIntensityImage ./ numSubSteps;

end

function Photon = propagatePhotons(Particle, PhotonDiffusion, NumberPhotons, ScatteringLengthUm, WallOffset, ChHeightUm, ChWidthUm, ChDepthUm)
% (!!! NOT FINISHED !!!) Update particles coordinates, diffusion etc

% Init photon struct
Photon.x = Particle.x;
Photon.y = Particle.y;
Photon.z = Particle.z;
Photon.Dx = PhotonDiffusion; % in [um2/s]
Photon.Dy = PhotonDiffusion; % in [um2/s]
Photon.Dz = PhotonDiffusion; % in [um2/s]

% Set time step of photon propagation
TimeStep = ScatteringLengthUm^2/6*PhotonDiffusion;

% Update particles position, update equation is diffusion --> r(t + dt) = r(t) + sqrt(2*D*dt)*gaussNoise
% In our case flow velocity is constant through channel => r(t + dt) = r(t) + sqrt(2*D*dt)*gaussNoise
for i = 1:NumberPhotons
        
    % Init photon position before the start of the propagation
    Photon.x = Particle.x;
    Photon.y = Particle.y;
    Photon.z = Particle.z;
    Photon.distance = 0.0;
    tmpDistance = 0.0;
    
    IsContinue = true;    
    while IsContinue == true % propagate while we hit Z observation plane or the wall (i.e. the photon escapes the box)
        % Generate the normal rnd numbers for the vectors of the position changes        
        rndNormal.x = randn(1);
        rndNormal.y = randn(1);
        rndNormal.z = randn(1);
        
        % Generate the discrete rnd numbers (-1,+1) mathcing mean and variance ot normal rnd numbers for the vectors of the position changes
        % rndNormal.x = 2*round(rand(1)) - 1;
        % rndNormal.y = 2*round(rand(1)) - 1;
        % rndNormal.z = 2*round(rand(1)) - 1;
        
        % Update current position --> origin of coordinate system is at X = 0, Y = 0, Z = 0, and the box is within the positive XYZ quadrant
        dx = sqrt(2*Photon.Dx*TimeStep)*rndNormal.x;
        dy = sqrt(2*Photon.Dy*TimeStep)*rndNormal.y; 
        dz = sqrt(2*Photon.Dz*TimeStep)*rndNormal.z;
        dr = sqrt(dx^2 + dy^2 + dz^2); % change in the propagated distance                
        tmpDistance = tmpDistance + dr; % overall propagated distance
        
        % Check for boundary collisions - the origin of coordinate system is at the top left corner of the window (x-axiss points to the right, y-axis points down)
        if Photon.z + dz < WallOffset.z % offset from the Z origin
            % Case we hit the observation plane (given by the offset)
            
        elseif Photon.z + dz > ChDepthUm
            % Photo lost --> exit while loop
            IsContinue = false;
        end
        
        if Photon.x + dx < WallOffset.x || Photon.x + dx > ChWidthUm
            % Photo lost --> exit while loop
            IsContinue = false;
        end

        if Photon.y + dy < WallOffset.y || Photon.y + dy > ChHeightUm
            % Photo lost --> exit while loop
            IsContinue = false;
        end
        
        % Update photon position
        Photon.x = Photon.x + dx;
        Photon.y = Photon.y + dy; 
        Photon.z = Photon.z + dz;
        Photon.distance = Photon.distance + dr; % overall propagated distance
    end
end

end

function rtrnImage = generateEFieldImage(Particles, ImageHeightPx, ImageWidthPx, RefIndexMedium, LaserWavelengthUm, PixelSizeUm)
% Generate Electrical field diffraction pattern in XY plane from particle positions by assuming spherical waves propagation

global glLSP;
RSCoreLookUpTbl = glLSP.RSCoreLookUpTbl;
observationPlaneXYZUm = glLSP.ObservationPlaneXYZUm;
stepSizeXYZUm = glLSP.StepSizeLookUpTbl;

numberParticles = length(Particles);
rtrnImage = zeros(ImageHeightPx, ImageWidthPx);
%k = 2*pi*RefIndexMedium/LaserWavelengthUm; % wave number

% Calc e-field image
for i = 1:numberParticles % loop thorough the particles
    dx = Particles(i).x - Particles(i).x0;
    dy = Particles(i).y - Particles(i).y0;
    dz = Particles(i).z - Particles(i).z0;
    dr = sqrt(dx^2 + dy^2 + dz^2); % length of the position change
         
    % Fine propagation to calc Electrical field diffraction pattern
    numSubSteps = floor(2*dr/LaserWavelengthUm + 1); % number of steps to walk between old and new position, each step moves a length equal half the wavelength
    dxFine = dx/numSubSteps; % fine dx step
    dyFine = dy/numSubSteps; % fine dy step
    dzFine = dz/numSubSteps; % fine dz step
    tmpParticlePosXUm = Particles(i).x0; % initial position
    tmpParticlePosYUm = Particles(i).y0; % initial position
    tmpParticlePosZUm = Particles(i).z0; % initial position    
    tmpImage = zeros(ImageHeightPx, ImageWidthPx); % reset
            
    % Calc El. field in the XY plane with Z = observation plane Z 
    for iStep = 1:numSubSteps
        Rz = tmpParticlePosZUm - observationPlaneXYZUm.Z; % observation plane is along Z
        iRz = ceil(Rz/stepSizeXYZUm);
        
        % Calc contribution of each particle-step to the image
%         for iPixRow = 1:ImageHeightPx
%             for iPixCol = 1:ImageWidthPx                                
%                 % Calc distance particle-pixel
%                 Rx = tmpParticlePosX - observationPlaneXYZUm.X(iPixRow, iPixCol);
%                 Ry = tmpParticlePosY - observationPlaneXYZUm.Y(iPixRow, iPixCol);
%                 %Rz = tmpParticlePosZ - observationPlaneXYZUm.Z;
%                 
%                 % Calc optical wavelet contribution to the given pixel --> using the core integrand from Rayleigh-Sommerfeld Diffraction Integral
%                 %Upix = calcRSDiffractionWavelet1(Particles(i).Ufield, Rx, Ry, Rz, k);
%                 %Upix = getRSCoreLookUpTblValue(Particles(i).Ufield, Rx, Ry, Rz);
%                 
%                 % Calc max distances
%                 R = sqrt(Rx^2 + Ry^2 + Rz^2); % max absolute distance source-point
%                 
%                 % Calc indexes for the lookup table value
%                 iR = ceil(R/stepSizeXYZUm);
%                 %iRz = ceil(Rz/stepSizeXYZUm);
%                                 
%                 % Assign intensity to the given pixel based on the particle position
%                 tmpImage(iPixRow, iPixCol) = tmpImage(iPixRow, iPixCol) + Upix(iR, iRz);
%             end
%         end
        
        % Calc contribution of each particle-step to the image - use Matrix form as much as possible to speedup calculations
        % Calc distance particle-pixel in Matrix form
        Rx = tmpParticlePosXUm - observationPlaneXYZUm.X;
        Ry = tmpParticlePosYUm - observationPlaneXYZUm.Y;
        %Rz = tmpParticlePosZUm - observationPlaneXYZUm.Z;

        % Calc max distances in Matrix form
        R = sqrt(Rx.^2 + Ry.^2 + Rz.^2); % max absolute distance source-point

        % Calc indexes for the lookup table value in Matrix form
        iR = ceil(R./stepSizeXYZUm);
        %iRz = ceil(Rz./stepSizeXYZUm);
        
        % Calc Ufield as a function of distance R matrix
        if Particles(i).Ufield == 1
            % Assumes Ufield = 1 => no multiplication necessary
            Upix = reshape(RSCoreLookUpTbl(iR(:), iRz), [ImageHeightPx, ImageWidthPx]); % precalc the RS core array for the given particle
        else
            Upix = Particles(i).Ufield .* reshape(RSCoreLookUpTbl(iR(:), iRz), [ImageHeightPx, ImageWidthPx]); % precalc the RS core array for the given particle
        end
        
        % Assign intensity to the given pixel based on the particle position --> non-vectorized form        
        %for iPixRow = 1:ImageHeightPx
        %   for iPixCol = 1:ImageWidthPx
        %       tmpImage(iPixRow, iPixCol) = tmpImage(iPixRow, iPixCol) + Upix(iR(iPixRow, iPixCol), iRz);
        %   end
        %end       
        
        % Assign intensity to the given pixel based on the particle position --> vectorized form
        tmpImage = tmpImage + Upix;
        
        % Update fine particle position --> go to the next position
        tmpParticlePosXUm = tmpParticlePosXUm + dxFine;
        tmpParticlePosYUm = tmpParticlePosYUm + dyFine;
        tmpParticlePosZUm = tmpParticlePosZUm + dzFine;
    end
    
    rtrnImage = rtrnImage + tmpImage./numSubSteps; % average over the num substeps    
end

end

function rtrnDiffractionImageXY = calcEfieldToIntensityImage(EImageXY, OutputField)
% Calculate diffraction/interference patttern from the input image (distribution of the electric field or the intensity)
% EImageXY = XY electric field distribution
% TypeOutput = 'e-field' (returns Electircal field distribution) | 'intnesity' (returns Intnesity distribution)

% Get dimenssions
[imgRows, imgCols] = size(EImageXY);
halfImgRows = round(imgRows/2);
halfImgCols = round(imgCols/2);

% Calc output image
switch(OutputField)
    case 'e-field'
        % Calc El. field image from the scattered electric field distribution
        rtrnDiffractionImageXY = EImageXY;
    case 'intensity'        
        % Calc intensity image from the scattered electric field distribution
        rtrnDiffractionImageXY = conj(EImageXY).*EImageXY;
        %rtrnDiffractionImageXY = rtrnDiffractionImageXY ./ max(rtrnDiffractionImageXY, [], 'all'); % normalize to 1
    otherwise
end

end

function rtrnDiffractionImageXY = diffractionPropagatedImage(EImageXY, OutputField)
% Calculate diffraction/interference patttern from the input image (distribution of the electric field or the intensity)
% EImageXY = XY electric field distribution
% TypeOutput = 'e-field' (returns Electircal field distribution) | 'intnesity' (returns Intnesity distribution)

% Get dimenssions
[imgRows, imgCols] = size(EImageXY);
halfImgRows = round(imgRows/2);
halfImgCols = round(imgCols/2);

% Assume the input image is a phase distribution instead of amplitude
%EImageXY = exp(-1i.*EImageXY);

% Zero padding to avoid ringing effects
%EImageXY = padarray(EImageXY, [imgRows, imgCols], 0, 'both');
EImageXY = padarray(EImageXY, [halfImgRows, halfImgCols], 0, 'both');

% Calc FFT of the input
%fft2EImageXY = fftshift(fft2(EImageXY));
fft2EImageXY = fft2(EImageXY);
%fft2EImageXY = fft2EImageXY(imgRows:(imgRows + imgRows - 1), imgCols:(imgCols + imgCols - 1)); % remove zero padding
fft2EImageXY = fft2EImageXY(halfImgRows:(halfImgRows + imgRows - 1), halfImgCols:(halfImgCols + imgCols - 1)); % remove zero padding
fft2EImageXY = fftshift(fft2EImageXY);

% Calc output image
switch(OutputField)
    case 'e-field'
        % Calc El. field image from the scattered electric field distribution
        rtrnDiffractionImageXY = fft2EImageXY;
    case 'intensity'        
        % Calc intensity image from the scattered electric field distribution
        rtrnDiffractionImageXY = conj(fft2EImageXY).*fft2EImageXY;
        rtrnDiffractionImageXY = rtrnDiffractionImageXY ./ max(rtrnDiffractionImageXY, [], 'all'); % normalize to 1
    otherwise
end

end

function [rtrnRow, rtrnCol] = getParticlePixelIndexes(XPos, YPos, PixelSizeUm)
% Find in which pixel the center of the PSF resides

rtrnRow = ceil(YPos/PixelSizeUm); % index of the pixel along the height of the image where the particle is
rtrnCol = ceil(XPos/PixelSizeUm); % index of the pixel along the height of the image where the particle is

end

function [rtrnX, rtrnY] = getPixelCenterXY(PixRow, PixCol, PixelSizeUm)
% Find the coordinates (physical) of the pixel center

rtrnX = PixCol*PixelSizeUm - 0.5*PixelSizeUm; % X coordinate of the pixel
rtrnY = PixRow*PixelSizeUm - 0.5*PixelSizeUm; % Y coordinate of the pixel

end

function rtrnPixelsCenterXY = getPixelsCenterXY(ImageHeightPx, ImageWidthPx, PixelSizeUm)
% Get all pixels coordinates in [um]

rtrnPixelsCenterXY.X = zeros(ImageHeightPx, ImageWidthPx);
rtrnPixelsCenterXY.Y = zeros(ImageHeightPx, ImageWidthPx);

for iPixRow = 1:ImageHeightPx
    for iPixCol = 1:ImageWidthPx
        % Find the coordinates (physical in um) of the pixel center given the pixel indexes
        [pixCenterXUm, pixCenterYUm] = getPixelCenterXY(iPixRow, iPixCol, PixelSizeUm);
        rtrnPixelsCenterXY.X(iPixRow, iPixCol) = pixCenterXUm;
        rtrnPixelsCenterXY.Y(iPixRow, iPixCol) = pixCenterYUm;
    end
end

end

function rtrnPixelValue = assignPixelValue(PsfType, PsfRadiusUm, PixelSizeUm, XPosParticleUm, YPosParticleUm, PixXCenterUm, PixYCenterUm)
% Assign intensity or el. field value to the current pixel based on the particle position and PSF/ePSF

% If pixel size is more than 1x bigger than PSF radius we need to use subpixel net (to satisfy Nyquist criterion)
PixelToPsfRatio = PixelSizeUm/PsfRadiusUm;
rtrnPixelValue = 0; % init value

if PixelToPsfRatio > 1
    % Case subpixel net to integrate Intensity/E-field
    numSubPixels = ceil(PixelToPsfRatio); % number of sub pixels along X or Y direction
    subPixelSize = PixelSizeUm/numSubPixels; % sub pixel length
    
    % Sample the PSF to obtain the intensity/e-field --> use square intengration
    for iX = 1:numSubPixels
        % Calc subpixel center using pixel center and subpixel number info
        subPixXCenterUm = PixXCenterUm - subPixelSize * (numSubPixels + 1 - 2*iX)/2; % coordinate of the subpixel with index iX along X
        X = XPosParticleUm - subPixXCenterUm; % x coordinate as seen from the center of the PSF        
        for iY = 1:numSubPixels
            subPixYCenterUm = PixYCenterUm - subPixelSize * (numSubPixels + 1 - 2*iY)/2; % coordinate of the subpixel with index iX along X
            Y = YPosParticleUm - subPixYCenterUm; % y coordinate as seen from the center of the PSF
            rtrnPixelValue = rtrnPixelValue + calc2DPSF(PsfType, PsfRadiusUm, X, Y); % intensity/e-field value in the XY point
        end
    end
    
    %rtrnPixelValue = subPixelSize^2 * rtrnPixelValue; % integrated in the XY point (subpixel integration)
else
    % Get intensity/e-field according the PSF (sampling is based on the center XY point of pixel)
    X = XPosParticleUm - PixXCenterUm; % x coordinate as seen from the center of the PSF
    Y = YPosParticleUm - PixYCenterUm; % y coordinate as seen from the center of the PSF
    rtrnPixelValue = calc2DPSF(PsfType, PsfRadiusUm, X, Y); % integrated intensity in the XY point
    %rtrnPixelValue = PixelSizeUm^2 * calc2DPSF(PsfType, PsfRadiusUm, X, Y); % integrated intensity/e-field in the XY point
end

end

function rtrnPsfWindow = setPsfWindow(XPos, YPos, PsfRadiusUm, PixelSizeUm, HeightPx, WidthPx)
% Calc pixel XY window indexes where we will calc the intensity

XPosPt = XPos/PixelSizeUm; % X particle coordinate in fractional [px]
YPosPt = YPos/PixelSizeUm; % Y particle coordinate in fractional [px]

PsfRadiusPt = PsfRadiusUm/PixelSizeUm; % fractional pixel PSF radius
PsfRangePt = 2*PsfRadiusPt; % range in terms of PSF radius where we will consider PSF intensity is strong enough to be detected

% Calc pixel window indexes
rtrnPsfWindow.minRow = ceil(YPosPt - PsfRangePt);
rtrnPsfWindow.minCol = ceil(XPosPt - PsfRangePt);
rtrnPsfWindow.maxRow = ceil(YPosPt + PsfRangePt);
rtrnPsfWindow.maxCol = ceil(XPosPt + PsfRangePt);

% Check range of indexes --> must be within bounds of the image
if rtrnPsfWindow.minRow < 1
    rtrnPsfWindow.minRow = 1;
end

if rtrnPsfWindow.maxRow > HeightPx
    rtrnPsfWindow.maxRow = HeightPx;
end

if rtrnPsfWindow.minCol < 1
    rtrnPsfWindow.minCol = 1;
end

if rtrnPsfWindow.maxCol > WidthPx
    rtrnPsfWindow.maxCol = WidthPx;
end

% Calc the size of the PSF window
rtrnPsfWindow.size = (rtrnPsfWindow.maxRow - rtrnPsfWindow.minRow + 1) * (rtrnPsfWindow.maxCol - rtrnPsfWindow.minCol + 1);

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

function rtrnU = calcRSDiffractionWavelet1(Ufield, Rx, Ry, Rz, k)
% Calculate Rayleigh-Sommerfeld diffraction wavelet (the core integrand)
% Rx = x distance source-(observation)point
% Ry = y distance source-(observation)point
% Rz = z distance source-(observation)point
% k = wave number (= 2*pi*n/lambda)

% Calc distance
absR = sqrt(Rx^2 + Ry^2 + Rz^2);

% Calc cos(n, r) - the angle between r and normal vector
cosnR = Rz/absR;

% Calc spherical wavelet contribution to the given pixel --> using the core integrand from Rayleigh-Sommerfeld Diffraction Integral
%rtrnU = -(1/(2*pi))* Ufield * exp(1i*k*absR) * cosnR / absR; % el. field contribution of the given particle to the given pixel for the guven substep
rtrnU = -(1/(2*pi))* Ufield * (1i*k - 1/absR) * exp(1i*k*absR) * cosnR / absR; % el. field contribution (+ nearfield term) of the given particle to the given pixel for the guven substep

end

function rtrnU = calcRSDiffractionWavelet2(Ufield, R, Rz, k)
% Calculate Rayleigh-Sommerfeld diffraction wavelet (the core integrand)
% R = xyz distance source-(observation)point
% Rz = z distance source-(observation)point
% k = wave number (= 2*pi*n/lambda)

% Calc distance
absR = R;

% Calc cos(n, r) - the angle between r and normal vector
cosnR = Rz/absR;

% Calc spherical wavelet contribution to the given pixel --> using the core integrand from Rayleigh-Sommerfeld Diffraction Integral
%rtrnU = -(1/(2*pi))* Ufield * exp(1i*k*absR) * cosnR / absR; % el. field contribution of the given particle to the given pixel for the guven substep
rtrnU = -(1/(2*pi))* Ufield * (1i*k - 1/absR) * exp(1i*k*absR) * cosnR / absR; % el. field contribution (+ nearfield term) of the given particle to the given pixel for the guven substep

end

function rtrnU = calcPlaneWavelet(Ufield, R, Rz, k)
% Calculate Plane Wave diffraction wavelet
% R = xyz distance source-(observation)point
% Rz = z distance source-(observation)point
% k = wave number (= 2*pi*n/lambda)

% Calc distance
absR = R;

% Calc cos(n, r) - the angle between r and normal vector
cosnR = Rz/absR;

% Calc spherical wavelet contribution to the given pixel --> using the plane wave approach
rtrnU = Ufield * exp(1i*k*absR) * cosnR; % el. field of plane wave of the given particle to the given pixel for the given substep

end

function rtrnRSCoreLookUpTbl = setRSCoreLookUpTbl(TypeWavelet, StepSizeUm, ObservationPlaneXYZUm, chHeightUm, chWidthUm, chDepthUm, RefIndexMedium, LaserWavelengthUm)
% Calculate the lookup table for the Rayleigh-Sommerfeld diffraction wavelet given the input box size, ref index and illumination wavelegth
% Once calculated the lookup table returns the value of the core RS integral for given R (distance) and Rz (z projection of R)

% Calc wavenumber
k = 2*pi*RefIndexMedium/LaserWavelengthUm;

% Calc max distances
maxRx = chHeightUm;
maxRy = chWidthUm;
maxRz = chDepthUm - ObservationPlaneXYZUm.Z;

maxR = sqrt(maxRx^2 + maxRy^2 + maxRz^2); % max absolute distance source-point

% Calc dimensions of the lookup table
lengthR = ceil(maxR/StepSizeUm);
lengthRz = ceil(maxRz/StepSizeUm);

% Amplitude of the field
Ufield = 1;

% Calc look up table for the given combination of R and Rz
rtrnRSCoreLookUpTbl = complex(zeros(lengthR, lengthRz));

switch(TypeWavelet)
    case {'plane-wavelet', 'plw'} % propagate light by Plane Wave propagator
        for iR = 1:lengthR
            R = iR*StepSizeUm;
            for iRz = 1:lengthRz
                Rz = iRz*StepSizeUm;
                rtrnRSCoreLookUpTbl(iR, iRz) = calcPlaneWavelet(Ufield, R, Rz, k);
            end
        end
    case {'rs-wavelet', 'rsw'} % propagate light by assuming Rayleigh-Sommerfeld Wave propagator
        for iR = 1:lengthR
            R = iR*StepSizeUm;
            for iRz = 1:lengthRz
                Rz = iRz*StepSizeUm;
                rtrnRSCoreLookUpTbl(iR, iRz) = calcRSDiffractionWavelet2(Ufield, R, Rz, k);
            end
        end
    otherwise
end

end

function rtrnU = getRSCoreLookUpTblValue(Ufield, Rx, Ry, Rz)
% Get the lookup table value for the Rayleigh-Sommerfeld diffraction wavelet given the input distances
% Once calculated the lookup table returns the value of the core RS integral for given R (distance) and Rz (z projection of R)

% Lookup table array
global glLSP;

% Calc max distances
R = sqrt(Rx^2 + Ry^2 + Rz^2); % max absolute distance source-point

% Calc indexes for the lookup table value
iR = ceil(R/glLSP.StepSizeLookUpTbl);
iRz = ceil(Rz/glLSP.StepSizeLookUpTbl);

% Get look up table value for the given combination of R and Rz
rtrnU = Ufield * glLSP.RSCoreLookUpTbl(iR, iRz);
 
end

function getParticlesSnapshot(particles, iZ, ArrayIndexes)
% Particles location snapshot of first and last positions

global glLSP;

if iZ == ArrayIndexes(1)
    glLSP.particlesSnapshot{1} = particles; % sanpshot of the 3D box
elseif iZ == ArrayIndexes(2)
    glLSP.particlesSnapshot{2} = particles; % snapshot of the 3D box
end

end

function saveParticlesSnapShot(BaseFileName, PixelSizeUm, HeightPx, WidthPx, DepthPx)
% Save particles location snapshot as 3D XYZ frames

global glLSP;

% Loop through the snapshots and save each snapshot as a 3D image volume
for iS = 1:length(glLSP.particlesSnapshot)
    particles = glLSP.particlesSnapshot{iS};
    image3DBoxXYZ = zeros(HeightPx, WidthPx, DepthPx);
    
    % Assign voxel according the calculated pixel indexes
    for iP = 1:length(particles)
        % Calc particle coordinates in px
        iCol = round(particles(iP).x ./PixelSizeUm);
        iRow = round(particles(iP).y ./PixelSizeUm);
        iZ   = round(particles(iP).z ./PixelSizeUm);
        
        % Check indexes are within range
        if iCol < 1
            iCol = 1;
        elseif iCol > WidthPx
            iCol = WidthPx;
        end
        
        if iRow < 1
            iRow = 1;
        elseif iRow > HeightPx
            iRow = HeightPx;
        end
           
        if iZ < 1
            iZ = 1;
        elseif iZ > DepthPx
            iZ = DepthPx;
        end
        
        % Assign voxel value
        image3DBoxXYZ(iRow, iCol, iZ) = image3DBoxXYZ(iRow, iCol, iZ) + 1.0;
    end
    
    % Save the 3D box - save data as image or video
    fprintf('\n');
    type3DStackItNormalization = 'global';
    outputFileType = 'tiff';
    currentFileName = sprintf('%s_box=%d', BaseFileName, iS);
    lsci_SaveToFrames(image3DBoxXYZ, currentFileName, outputFileType, type3DStackItNormalization);
end
    
end