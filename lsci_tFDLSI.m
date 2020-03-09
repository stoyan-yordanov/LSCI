function lsci_tFDLSI(InputFile, NumericalMethod, StartFrame, EndFrame, PixelXYZ, ZWindowSizeFrames, CamExposureTime, FrameRate, WavelengthUm, NA, Magnification)
% Read video or images, calcs decorelation time and velocity map by Temporal Frequency Domain Laser Speckle Imaging and save it as multi page tiff file.
% InputFile = file name of the input data (if empty brings command line file dialog) - supports avi (video) | mj2 (video Motion Jpeg 2000) | tiff (multipage)
% NumericalMethod = calc Ct(tau) using 'fft' (Fast Fourier Transform) or 'xcov' (built in matlab (auto)-covariance function)
% Process between StartFrame and EndFrame frames.
% PixelXYZ = [X, Y, Z] (coordinate of the point where we show statistics (tc and V)
% ZWindowSizeFrames = 500, 1000 etc (pixel size of the Z sliding window to calc Ct(tau) - the autocovariance)
% CamExposureTime = 250e-6 [sec] etc (cam exposure time in sec.)
% FrameRate = frames per second (sampling frequency)
% WavelengthUm = wavelength of illumination in [um]
% NA = numerical aperture of the imaging system
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
dataLSP.Autocovariance = []; % LSP Autocovariance --> represents CtTau
dataLSP.CorrelationTime = []; % LSP Correlation Time --> represents Tc
dataLSP.Velocity = []; % Velocity --> represents V

% Write frames to multipage tiff file
fprintf('\nStart calculating Temporal Frequency Domain Laser Speckle... \n'); % show progress

% Calc Autocovariance Map --> Ct(tau)
dataLSP.Autocovariance = calcLSPAutocovarianceMap(inXYZFrames, ZWindowSizeFrames, NumericalMethod, FrameRate);

% Calc Corraletion Time Map --> Tc
dataLSP.CorrelationTime = calcLSPCorrelationTimeMap(dataLSP.Autocovariance);

% Calc Velocity Map --> vs/v0/vd
dataLSP.Velocity = calcLSPVelocityMap(dataLSP.Autocovariance, Magnification, WavelengthUm, NA);

% Save result for Laser Speckle Decorelation time as tiff
tiffFileNamePathFDLStc = fullfile(filePath, [fileName '_tFDLS-tc' '.tiff']); % Assemble tiff file name for Laser Speckle Autocovariance Ct(tau) or tc (de-correlation time)
maxTc = max(dataLSP.CorrelationTime.value, [], 'all');
minTc = min(dataLSP.CorrelationTime.value, [], 'all');
for i = 1:size(dataLSP.CorrelationTime.value, 3)
    % % coefficient to convert to 8 bit integer
    
    frameTc = dataLSP.CorrelationTime.value(:, :, i)./maxTc; % normalize to 1
    cInt = 2^16 - 1; % cInt = 255; % coefficient to convert to 8 or 16 bit integer
    frameTc = uint16(cInt*frameTc); % convert to 8 or 16 bit depth
    
    imwrite(frameTc, tiffFileNamePathFDLStc, 'tiff', 'Compression', 'packbits', 'WriteMode', 'append'); % save Laser Speckle Decorrelation Time    
end

% Set base file names
tiffFileNamePathVs = fullfile(filePath, [fileName '_tFDLS-vs' '.tiff']); % Assemble tiff file name for Laser Speckle Velocity
tiffFileNamePathV0 = fullfile(filePath, [fileName '_tFDLS-v0' '.tiff']); % Assemble tiff file name for Laser Speckle Velocity
tiffFileNamePathVd = fullfile(filePath, [fileName '_tFDLS-vd' '.tiff']); % Assemble tiff file name for Laser Speckle Velocity

% Save result for Laser Speckle Velocity map as tiff
maxVs = max(dataLSP.Velocity.vs.value, [], 'all'); % max velocity in Vs (single flow) dim
maxV0 = max(dataLSP.Velocity.v0.value, [], 'all'); % max velocity in V0 (diffusion model) dim
maxVd = max(dataLSP.Velocity.vd.value, [], 'all'); % max velocity in Vd (diffusion model) dim
maxV = max([maxVs, maxV0, maxVd], [], 'all'); % max velocity in all dim

for i = 1:size(dataLSP.Velocity.vs.value, 3)        
    % Prepare to save it as an image(s)
    cInt = 2^16 - 1; % cInt = 255; % coefficient to convert to 8 or 16 bit integer
    frameVs = uint16(cInt*(dataLSP.Velocity.vs.value(:, :, i)./maxVs)); % convert to 8 or 16 bit depth
    frameV0 = uint16(cInt*(dataLSP.Velocity.v0.value(:, :, i)./maxV0)); % convert to 8 or 16 bit depth
    frameVd = uint16(cInt*(dataLSP.Velocity.vd.value(:, :, i)./maxVd)); % convert to 8 or 16 bit depth
    
    imwrite(frameVs, tiffFileNamePathVs, 'tiff', 'Compression', 'packbits', 'WriteMode', 'append'); % save Laser Speckle Velocity - single flow (no difusion model)
    imwrite(frameV0, tiffFileNamePathV0, 'tiff', 'Compression', 'packbits', 'WriteMode', 'append'); % save Laser Speckle Velocity - directed flow (diffusion model)
    imwrite(frameVd, tiffFileNamePathVd, 'tiff', 'Compression', 'packbits', 'WriteMode', 'append'); % save Laser Speckle Velocity - flow due to diffusion component (diffusion model)
end

% SHow progress
elapsedTime = toc(startTime);

fprintf('\n\nEnd of processing --> Start Frame = %d, End Frame = %d\n', StartFrame, EndFrame); % show progress
fprintf('Statistics Correlation Time --> (All) Max Tc = %.3g [s], Min Tc (all) = %.3g [s]\n', maxTc, minTc); % show stat
fprintf('Statistics Correlation Time Pixel[%d, %d, %d] --> Tc = %.3g ± %.3g [s]\n', pixY, pixX, pixZ, dataLSP.CorrelationTime.value(pixX, pixY, pixZ), (dataLSP.CorrelationTime.upper(pixX, pixY, pixZ) - dataLSP.CorrelationTime.lower(pixX, pixY, pixZ))/2); % show stat
fprintf('Statistics Velocity --> Max V (all) = %.3f [mm/s], Max. Vs/V0/Vd = %.3f/%.3f/%.3f [mm/s]\n', maxV, maxVs, maxV0, maxVd); % show stat
fprintf('Statistics Velocity Pixel[%d, %d, %d] --> Vs = %.3f ± %.3g [mm/s], V0 = %.3f ± %.3g [mm/s], Vd = %.3f ± %.3g [mm/s]\n', pixY, pixX, pixZ,...
    dataLSP.Velocity.vs.value(pixX, pixY, pixZ), (dataLSP.Velocity.vs.upper(pixX, pixY, pixZ) + dataLSP.Velocity.vs.lower(pixX, pixY, pixZ))/2,...
    dataLSP.Velocity.v0.value(pixX, pixY, pixZ), (dataLSP.Velocity.v0.upper(pixX, pixY, pixZ) + dataLSP.Velocity.v0.lower(pixX, pixY, pixZ))/2,...
    dataLSP.Velocity.vd.value(pixX, pixY, pixZ), (dataLSP.Velocity.vd.upper(pixX, pixY, pixZ) + dataLSP.Velocity.vd.lower(pixX, pixY, pixZ))/2); % show stat
fprintf('Processing time = %f [sec]\n\n', elapsedTime);

end

function rtrnCtTau = calcLSPAutocovarianceMap(InXYZFrames, ZWindowSizeFrames, NumericalMethod, FrameRate)
% Calc decorelation time and velocity map by Temporal Frequency Domain Laser Speckle Imaging
% Numerical algorithms take ideas from the following papers:
% Hao Li et al, "Directly measuring absolute flow speed by frequency-domain laser speckle imaging", DOI link: https://doi.org/10.1364/OE.22.021079

% Calc autocovariance Ct(tau) map
rtrnCtTau = tFDLSIDirect(InXYZFrames, ZWindowSizeFrames, NumericalMethod, FrameRate); % calc autocovariance for each pixel in XY, num curzes Z, T = Tau values for given curve

end

function rtrnCtTau = tFDLSIDirect(InXYZFrames, ZWindowSizeFrames, NumericalMethod, FrameRate)
% Calculate autocovariance (for each pixel in XY) of intensity signal by FFT or xcov() methods

% Calc Ct(tau), i.e. the auto-covariance
switch(NumericalMethod)
    case 'fft' % Calc using FFT
        rtrnCtTau = tAutoCovarianceFFT(InXYZFrames, ZWindowSizeFrames, FrameRate);        
    case 'xcov' % calc using buit in matalb covariance xcov() function
        rtrnCtTau = tAutoCovarianceXcov(InXYZFrames, ZWindowSizeFrames, FrameRate);
    otherwise
        frintf('\n\nUnsupported numerical method: %s\', NumericalMethod);
        error('Exit due to the above error!');
end

end

function rtrnCtTau = tAutoCovarianceFFT(InXYZFrames, ZWindowSizeFrames, FrameRate)
% Calculate autocovariance (for each pixel in XY) of intensity signal by Fourier Transform (see eq 1, 3 and 4 in the paper below)
% Hao Li et al, "Directly measuring absolute flow speed by frequency-domain laser speckle imaging", DOI link: https://doi.org/10.1364/OE.22.021079

[lengthX, lengthY, lengthZ] = size(InXYZFrames);
lengthT = floor(lengthZ/ZWindowSizeFrames); % how many times the z window fits in all frames

% Pre-allocate Ct(tau) --> X = pixel X, Y = pixel Y, lengthT = num curves Ct(tau), ZWindowSizeFrames = num points in each Ct(tau) curve
rtrnCtTau = struct([]);
rtrnCtTau(1).Ct = zeros(lengthX, lengthY, ZWindowSizeFrames); % autocovariance trace in XYZ per given time point window ZWindowSizeFrames
rtrnCtTau(1).Tau = zeros(ZWindowSizeFrames, 1); % tau (time) trace

frameTime = 1/FrameRate;

% Calc Ct(tau) by FFT
fprintf('\nProgress, calc Ct(tau) by FFT: 000.0 [%%] | 00000.0 [sec]');
iZBegin = 1;
%workerThreads = 4;
%parfor (iZ = 1:fitLengthZ, workerThreads) % parallel for-loop through frames
for iT = 1:lengthT % loop as many times as ZWindowSizeFrames fits in all available frames
    startTime = tic;
    % Calc autocovariance Ct(tau) = <I°(t)I(t+tau)> - <I>^2
    for iX = 1:lengthX % loop through image height
        for iY = 1:lengthY % loop through image width
            subFrame = InXYZFrames(iX, iY, iZBegin:(iZBegin+ZWindowSizeFrames-1)); % extract subframe given by the window size
            subFrame = subFrame(:);
            meanIntensity = sum(subFrame, 'all')/ZWindowSizeFrames; % mean intensity
            subFrameZP = [subFrame - meanIntensity; zeros(ZWindowSizeFrames)]; % add zero padding to properly calc the autocovariance
            
            Iw = fft(subFrameZP); % calc the Fourier Transform of the input intensity along temporal direction I(w) = F(I(t)) = 1/2Pi*integral(I(t)*exp(-I*w*t)*dt)
            sqrIw = abs(Iw).^2; % the square amplitude of the power spectrum of the input signal
            CtTau = ifft(sqrIw); % calc Ct(tau) as FT of the square of the power spectrum of the input signal (plus remove the contribution of the DC term)
            CtTau = CtTau(1:ZWindowSizeFrames); % select only the non-zero padded part
            
            CtTauNormalized = CtTau./CtTau(1); % normalize to remove the dependence on the illumination intensity
            rtrnCtTau(iT).Ct(iX, iY, 1:ZWindowSizeFrames) = CtTauNormalized; % the normalized autocovariance for the given pixel XY and window along Z (each curve gets its own index iZ)
        end
    end
    rtrnCtTau(iT).Tau = (0:ZWindowSizeFrames - 1)*frameTime; % calc tau trace
    iZBegin = iT*ZWindowSizeFrames;
    elapsedTime = toc(startTime);
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'); % delete previous progress line
    fprintf('%05.1f [%%] | %07.1f [sec]', (iT/lengthT)*100, (lengthT-iT)*elapsedTime);
end

end

function rtrnCtTau = tAutoCovarianceXcov(InXYZFrames, ZWindowSizeFrames, FrameRate)
% Calculate autocovariance (for each pixel in XY) of intensity signal by matlab xcov()

[lengthX, lengthY, lengthZ] = size(InXYZFrames);
lengthT = floor(lengthZ/ZWindowSizeFrames); % how many times the z window fits in all frames

% Pre-allocate Ct(tau) --> X = pixel X, Y = pixel Y, lengthT = num curves Ct(tau), ZWindowSizeFrames = num points in each Ct(tau) curve
rtrnCtTau = struct([]);
rtrnCtTau(1).Ct = zeros(lengthX, lengthY, ZWindowSizeFrames); % autocovariance trace in XYZ per given time point window ZWindowSizeFrames
rtrnCtTau(1).Tau = zeros(ZWindowSizeFrames, 1); % tau (time) trace

frameTime = 1/FrameRate;

% Calc Ct(tau) by xcov()
fprintf('\nProgress, calc Ct(tau) by xcov(): 000.0 [%%] | 00000.0 [sec]');
iZBegin = 1;
%workerThreads = 4;
%parfor (iZ = 1:fitLengthZ, workerThreads) % parallel for-loop through frames
for iT = 1:lengthT % loop as many times as ZWindowSizeFrames fits in all available frames
    startTime = tic;    
    % Calc autocovariance Ct(tau) = <I°(t)I(t+tau)> - <I>^2
    for iX = 1:lengthX % loop through image height
        for iY = 1:lengthY % loop through image width
            subFrame = InXYZFrames(iX, iY, iZBegin:(iZBegin+ZWindowSizeFrames-1)); % extract subframe given by the window size
            CtTau = xcov(subFrame(:)); % calc Ct(tau)
            CtTau = CtTau(ZWindowSizeFrames:end); % take the range [0, max]
            CtTauNormalized = CtTau./max(CtTau); % normalize to remove the dependence on the illumination intensity
            rtrnCtTau(iT).Ct(iX, iY, 1:ZWindowSizeFrames) = CtTauNormalized; % the normalized autocovariance for the given pixel XY and window along Z (each curve gets its own index iZ)
        end
    end
    rtrnCtTau(iT).Tau = (0:ZWindowSizeFrames - 1)*frameTime; % calc tau trace
    iZBegin = iT*ZWindowSizeFrames;
    elapsedTime = toc(startTime);
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'); % delete previous progress line
    fprintf('%05.1f [%%] | %07.1f [sec]', (iT/lengthT)*100, (lengthT-iT)*elapsedTime);
end

end

function rtrnXYZCorrelationTime = calcLSPCorrelationTimeMap(LSPAutocovariance)
% Calc (de)corelation time map from Ct(tau) map (eq. 9 use fitting directly tc, lo = M*v*tc)
% Hao Li et al, "Directly measuring absolute flow speed by frequency-domain laser speckle imaging", DOI link: https://doi.org/10.1364/OE.22.021079

[lengthT] = length(LSPAutocovariance); % num global time points
[lengthX, lengthY, lengthZ] = size(LSPAutocovariance(1).Ct);

% The returned structure
rtrnXYZCorrelationTime = struct();
rtrnXYZCorrelationTime.value = zeros(lengthX, lengthY, lengthT); % correlation time
rtrnXYZCorrelationTime.lower = zeros(lengthX, lengthY, lengthT); % lower confidence interval of tc
rtrnXYZCorrelationTime.upper = zeros(lengthX, lengthY, lengthT); % upper confidence interval of tc
rtrnXYZCorrelationTime.unit = '[s]'; % tc unit

% Set common fit options
fitOptions = fitoptions(...
    'Normalize', 'off',...
    'Exclude', [],...
    'Weights', [],...
    'Method', 'NonlinearLeastSquares',...
    'Robust', 'off',...
    'StartPoint', [],...
    'Lower', [],...
    'Upper', [],...
    'Algorithm', 'Trust-Region',...
    'DiffMinChange', 1.0000e-08,...
    'DiffMaxChange', 0.1000,...
    'Display', 'off',...
    'MaxFunEvals', 20000,...
    'MaxIter', 5000,...
    'TolFun', 1.0000e-09,...
    'TolX', 1.0000e-09);

% Fit temporal autocovariance with exponential fit function --> Ct(tau) = exp(-tau^2/tc^2), tc = lo/M*v
CtTcTauFitFunc = @(tc, x) exp(-(x./tc).^2); % tc = coefficient to be determined, Tau = x independent variable
fitOptions.StartPoint = 1e-3; % v --> starting decorelation time tc in [s]
fitOptions.Lower = 0; % tc --> lower bound of tc in [s]

fprintf('\nProgress, calc tc: 000.0 [%%] | 00000.0 [sec]');
for iT = 1:lengthT % loop through the num of curves in Z
    Tau = LSPAutocovariance(iT).Tau;
    Tau = Tau(:);    
    for iX = 1:lengthX % loop through pixels along X
        startTime = tic;
        for iY = 1:lengthY % loop through pixels along Y
            CtTau = LSPAutocovariance(iT).Ct(iX, iY, :);
            CtTau = CtTau(:);
            if sum(isnan(CtTau))> 0 % check for NaNs
                % Case NaN found
                rtrnXYZCorrelationTime.value(iX, iY, iT) = 0; % [s] - decorelation time tc
            else
                % Case no NaN --> we can do the fit                
                [fitObjectCtTcTau, gofCtTcTau, outputCtTcTau] = fit(Tau, CtTau, CtTcTauFitFunc, fitOptions); % Tau = x, CtTau = y, CtTcTauFitFunc = autocovariance fit function
                
                vci = confint(fitObjectCtTcTau); % 95% confidence interaval in [s]
                rtrnXYZCorrelationTime.value(iX, iY, iT) = fitObjectCtTcTau.tc; % [s] - decorelation time tc
                rtrnXYZCorrelationTime.lower(iX, iY, iT) = vci(1); % [s] - tc lower bound
                rtrnXYZCorrelationTime.upper(iX, iY, iT) = vci(2); % [s] - tc upper bound
            end
        end
        elapsedTime = toc(startTime);
        progressPercentage = (((iT-1)*lengthT + iX)/(lengthX*lengthT))*100;
        remainingTime = ((lengthT*lengthX) - lengthT*(iT-1) - iX)*elapsedTime;
        fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'); % delete previous progress line        
        fprintf('%05.1f [%%] | %07.1f [sec]', progressPercentage, remainingTime);
    end
end

end

function rtrnXYZVelocity = calcLSPVelocityMap(LSPAutocovariance, Magnification, WavelengthUm, NA)
% Calc velocity map from Ct(tau) map (eq. 9 and eq. 13)
% Hao Li et al, "Directly measuring absolute flow speed by frequency-domain laser speckle imaging", DOI link: https://doi.org/10.1364/OE.22.021079

[lengthT] = length(LSPAutocovariance); % num global time points
[lengthX, lengthY, lengthZ] = size(LSPAutocovariance(1).Ct);

% The returned structure
rtrnXYZVelocity = struct();

rtrnXYZVelocity.vs.value = zeros(lengthX, lengthY, lengthT); % single velocity map per time point (pure flow model)
rtrnXYZVelocity.vs.lower = zeros(lengthX, lengthY, lengthT); % lower confidence interval of vs
rtrnXYZVelocity.vs.upper = zeros(lengthX, lengthY, lengthT); % upper confidence interval of vs
rtrnXYZVelocity.vs.unit = '[mm/s]'; % vs unit

rtrnXYZVelocity.v0.value = zeros(lengthX, lengthY, lengthT); % mean velocity map per time point (diffusion + flow model)
rtrnXYZVelocity.v0.lower = zeros(lengthX, lengthY, lengthT); % lower confidence interval of v0
rtrnXYZVelocity.v0.upper = zeros(lengthX, lengthY, lengthT); % upper confidence interval of v0
rtrnXYZVelocity.v0.unit = '[mm/s]'; % v0 unit

rtrnXYZVelocity.vd.value = zeros(lengthX, lengthY, lengthT); % rms velocity map per time point (difusion + flow model)
rtrnXYZVelocity.vd.lower = zeros(lengthX, lengthY, lengthT); % lower confidence interval of vd
rtrnXYZVelocity.vd.upper = zeros(lengthX, lengthY, lengthT); % upper confidence interval of vd
rtrnXYZVelocity.vd.unit = '[mm/s]'; % vd unit

% Set common fit options
fitOptionsVs = fitoptions(...
    'Normalize', 'off',...
    'Exclude', [],...
    'Weights', [],...
    'Method', 'NonlinearLeastSquares',...
    'Robust', 'off',...
    'StartPoint', [],...
    'Lower', [],...
    'Upper', [],...
    'Algorithm', 'Trust-Region',...
    'DiffMinChange', 1.0000e-08,...
    'DiffMaxChange', 0.1000,...
    'Display', 'off',...
    'MaxFunEvals', 20000,...
    'MaxIter', 5000,...
    'TolFun', 1.0000e-09,...
    'TolX', 1.0000e-09);

% Set common fit options
fitOptionsV0Vd = fitoptions(...
    'Normalize', 'off',...
    'Exclude', [],...
    'Weights', [],...
    'Method', 'NonlinearLeastSquares',...
    'Robust', 'off',...
    'StartPoint', [],...
    'Lower', [],...
    'Upper', [],...
    'Algorithm', 'Trust-Region',...
    'DiffMinChange', 1.0000e-08,...
    'DiffMaxChange', 0.1000,...
    'Display', 'off',...
    'MaxFunEvals', 20000,...
    'MaxIter', 5000,...
    'TolFun', 1.0000e-09,...
    'TolX', 1.0000e-09);

% Fit temporal autocovariance with single flow velocity function (no difusion)
CtVsTauFitFunc = @(vs, x) calcTheoryCtVTau(x, vs, Magnification, WavelengthUm, NA); % vs = velocity coefficient to be determined, Tau = x independent variable
fitOptionsVs.StartPoint = 100; % v --> starting velocity in [um/s]
fitOptionsVs.Lower = 0; % v --> lower bound of velocity in [um/s]

% Fit temporal autocovariance with mean flow velocity and diffusion
CtV0VdTauFitFunc = @(v0, vd, x) calcTheoryCtV0VdTau(x, v0, vd, Magnification, WavelengthUm, NA); % v0, vd = velociety coefficient to be determined, Tau = x independent variable
fitOptionsV0Vd.StartPoint = [100, 100]; % [v0, vd] --> starting velocities in [um/s]
fitOptionsV0Vd.Lower = [0, 0]; % [v0, vd] --> lower bound of velocities in [um/s]

fprintf('\nProgress, calc vs/v0/vd: 000.0 [%%] | 00000.0 [sec]');
for iT = 1:lengthT % loop through the num of curves in T
    Tau = LSPAutocovariance(iT).Tau;
    Tau = Tau(:);
    for iX = 1:lengthX % loop through pixels along X
        startTime = tic;
        for iY = 1:lengthY % loop through pixels along Y
            CtTau = LSPAutocovariance(iT).Ct(iX, iY, :);
            CtTau = CtTau(:);
            if sum(isnan(CtTau))> 0 % check for NaNs
                % Case NaN found
                rtrnXYZVelocity.vs.value(iX, iY, iT) = 0; % [mm/s] - directed single flow velocity (no diffusion)
                rtrnXYZVelocity.v0.value(iX, iY, iT) = 0; % [mm/s] - directed velocity (diffusion + flow)
                rtrnXYZVelocity.vd.value(iX, iY, iT) = 0; % [mm/s] - rms velocity (diffusion + flow)
            else
                % Case no NaN --> we can do the fit
                
                % Fit for vs
                [fitObjectCtV0Tau, gofCtV0Tau, outputCtV0Tau] = fit(Tau, CtTau, CtVsTauFitFunc, fitOptionsVs); % Tau = x, CtTau = y, CtV0TauFitFunc = singl velocity fit function
                
                vci = confint(fitObjectCtV0Tau)./1000; % 95% confidence interaval in [mm/s]
                rtrnXYZVelocity.vs.value(iX, iY, iT) = fitObjectCtV0Tau.vs/1000; % [mm/s] - directed single flow velocity (no diffusion)
                rtrnXYZVelocity.vs.lower(iX, iY, iT) = vci(1); % vs lower bound
                rtrnXYZVelocity.vs.upper(iX, iY, iT) = vci(2); % vs upper bound
                
                % Fit for v0 and vd
                [fitObjectCtV0VdTau, gofCtV0VdTau, outputCtV0VdTau] = fit(Tau, CtTau, CtV0VdTauFitFunc, fitOptionsV0Vd); % Tau = x, CtTau = y, CtV0VdTauFitFunc = diffusion + flow fit function
               
                vci = confint(fitObjectCtV0VdTau)./1000; % 95% confidence interaval in [mm/s]
                rtrnXYZVelocity.v0.value(iX, iY, iT) = fitObjectCtV0VdTau.v0/1000; % [mm/s] - directed single flow mean velocity (with diffusion)
                rtrnXYZVelocity.v0.lower(iX, iY, iT) = vci(1, 1); % vs lower bound
                rtrnXYZVelocity.v0.upper(iX, iY, iT) = vci(2, 1); % vs upper bound
                
                rtrnXYZVelocity.vd.value(iX, iY, iT) = fitObjectCtV0VdTau.vd/1000; % [mm/s] - root mean squared velocity (due to diffusion)
                rtrnXYZVelocity.vd.lower(iX, iY, iT) = vci(1, 2); % vs lower bound
                rtrnXYZVelocity.vd.upper(iX, iY, iT) = vci(2, 2); % vs upper bound
            end            
        end
        elapsedTime = toc(startTime);
        progressPercentage = (((iT-1)*lengthT + iX)/(lengthX*lengthT))*100;
        remainingTime = ((lengthT*lengthX) - lengthT*(iT-1) - iX)*elapsedTime;
        fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'); % delete previous progress line
        fprintf('%05.1f [%%] | %07.1f [sec]', progressPercentage, remainingTime);
    end
end

end

function rtrnCt = calcTheoryCtVTau(Tau, Velocity, Magnification, WavelengthUm, NA)
% Calculate Ct(tau, v) with no diffusion (i.e. velocity >> diffusion), Ct(tau, v) = exp(-(M*v*tau)^2/len0^2), l0 = decorelation length (see eq. 9 in the paepr below)
% Hao Li et al, "Directly measuring absolute flow speed by frequency-domain laser speckle imaging", DOI link: https://doi.org/10.1364/OE.22.021079
% Velocity = direted flow velocity [um/s]

% Calc decorelation length
len0 = 0.41*Magnification*WavelengthUm/NA;

% Calc Ct(tau, v)
rtrnCt = exp(-(Magnification.*Velocity.*Tau).^2./len0^2);

end

function rtrnCt = calcTheoryCtV0VdTau(Tau, V0, Vd, Magnification, WavelengthUm, NA)
% Calculate Ct(tau, v) with diffusion, i.e. Ct(tau, v0, vd), len0 = decorelation length (see eq. 13 in the paper below)
% Hao Li et al, "Directly measuring absolute flow speed by frequency-domain laser speckle imaging", DOI link: https://doi.org/10.1364/OE.22.021079
% V0 = mean velocity (directed velocity due to flow) [um/s]
% Vd = root mean square velocity (due to diffusion) [um/s]

% Calc decorelation length
len0 = 0.41*Magnification*WavelengthUm/NA;

% Calc Ct(tau, v0, vd)
Ct1 = exp(-(Magnification.*V0.*Tau).^2./(len0^2 + (Magnification.*Vd.*Tau).^2));
Ct2 = len0^3./(len0^2 + (Magnification.*Vd.*Tau).^2).^1.5 + (2.*Magnification.^4.*V0.^2.*Vd.^2.*len0.*Tau.^4)./(len0^2 + (Magnification.*Vd.*Tau).^2).^2.5;

rtrnCt = Ct1.*Ct2;

end