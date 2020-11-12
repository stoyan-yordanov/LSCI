function lsci_meLASCA(InputFile, StartFrame, EndFrame, CamExposureTime, MultiExposureAlgorithm, MultiExposureGroupFrames, FrameRate, PixelXY, XYWindowSizePx, WavelengthUm, NA, FitMethodForVelocity)
% Read input (video or images), analyse laser speckle by Multiexposure Laser Speckle Contrast Analysis
% Algorithm is based on the following paper: 
% Oliver Thomson et al, "Tissue perfusion measurements: multiple-exposure laser speckle analysis generates laser Doppler–like spectra", DOI: https://doi.org/10.1117/1.3400721
% Function parameters:
% InputFile = file name of the input data (if empty brings command line file dialog) - supports avi (video) | mj2 (video Motion Jpeg 2000) | tiff (multipage)
% Process between StartFrame and EndFrame frames.
% CamExposureTime = 250e-6 [sec] etc (cam exposure time in sec.)
% MultiExposureAlgorithm = the increase of exposure time with frame number - 'off' (new dt = exposure time) | 'lin' (new dt = dt + exposure time) | 'pow2' (new dt = 2*dt)
% MultiExposureGroupFrames = 'on' (group frames per multi-exposure time to increase accuracy) | 'off' (no grouping of frames per multi-exposure time --> much less accurate)
% FrameRate = frames per second
% PixelXY = [X, Y] or [cols, rows] (coordinate of the point where we calc K(T), Ct(tau), Velocity etc)
% XYWindowSizePx = [X, Y] or [cols, rows] size in pixels over which we calc K (Laser Speckle Contrast)
% CamPixelSizeUm = physical sze of a pixel in [um]
% Magnification = magnification of the optical system
% WavelengthUm = wavelength of the illumination light in [um]
% NA = numerical aperture
% FitMethodForVelocity = fit method to extract velocity --> 'fit-autocovariance' (fit Autocovariance vs Tau curve) | 'fit-lsp-contrast' (fit LSP Contrast vs Exposure Time curve)


% Precalc
%PixelSize = CamPixelSizeUm/Magnification; % pixel size in image plane in [um] (it will depend on the magnification)

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

% Extract substack --> we analyze a substack around the given pixels coordinates
halfPixelWindowX = floor(XYWindowSizePx(2)/2);
halfPixelWindowY = floor(XYWindowSizePx(1)/2);
inXYZFrames = inXYZFrames((PixelXY(2) - halfPixelWindowX + 1):(PixelXY(2) + halfPixelWindowX), (PixelXY(1) - halfPixelWindowY + 1):(PixelXY(1) + halfPixelWindowY), :); % get XY-Z 3D slice

% The structure that will store the results of the processing
dataLSP = struct();
dataLSP.Contrast = struct([]); % LSP Contrast = Contrast(i = curve ndex, i.e. Z-GlobalTimeResolution).globalTime .Time(j) .K(j)/.Kfit(j) --> represents K(j) vs Time(j) curve (per chosen pixel)
dataLSP.Autocovariance = struct([]); % LSP Autocovariance = Autocovariance(i = curve ndex, i.e. Z-GlobalTimeResolution).globalTime .Tau(j2) .Ct(j2)/.Ctvsfit(j2)/.Ctv0vdfit(j2) --> represents Ct(j2) vs Tau(j2) curve (per chosen pixel)
dataLSP.PSD = struct([]); % Power Spectrum Density = PSD(i = curve ndex, i.e. Z-GlobalTimeResolution).globalTime .Freq(j2) .PsdAmplitude(j2) --> represents PsdAmplitude(j2) vs Freq(j2) curve (per chosen pixel)
dataLSP.Velocity = struct([]); % Velocity(i = curve ndex, i.e. Z-GlobalTimeResolution).globalTime .vs/v0/vd --> represents vs/v0/vd vs GlobalTime (per chosen pixel)

% Start processing
fprintf('\nStart Multiexposure Laser Speckle Analysis... \n'); % show progress

% Calc Laser Speckle Contrast Map (K)
dataLSP.Contrast = calcLSPContrastMap(inXYZFrames, CamExposureTime, FrameRate, MultiExposureAlgorithm, MultiExposureGroupFrames);

% Calc Autocovariance Map (Ct(tau))
dataLSP.Autocovariance = calcLSPAutocovarianceMap(dataLSP.Contrast);

% Calc PSD (Power Spectrum Density = abs(FFT)^2 = fft(Ct(tau)) ) from Autocovariance Map (Ct(tau))
dataLSP.PSD = calcPsdMap(dataLSP.Autocovariance);

% Calc Velocity Map:
% Method 1 - Ct(tau) fitting:(Vs - single velosity; V0 - single velocity + Vd - diffusion) by fitting the Autocovariance Map (Ct(tau)) with model function
% Method 2 - K vs Exposure time fitting
dataLSP = calcVelocityMap(dataLSP, FitMethodForVelocity, WavelengthUm, NA);

% Save processed data
saveLspData(dataLSP, InputFile, PixelXY, XYWindowSizePx, CamExposureTime, FrameRate, WavelengthUm, NA, MultiExposureAlgorithm, MultiExposureGroupFrames, FitMethodForVelocity);

% Show processed data
%showProcessedData();

% Print progress
elapsedTime = toc(startTime);

fprintf('\n\nEnd of processing --> Start Frame = %d, End Frame = %d\n', StartFrame, EndFrame); % show progress
%fprintf('Statistics --> V (single flow) = %.3f ± %.3f [mm/s]\n', vMmps.value, (vMmps.upper - vMmps.lower)/2); % show stat
%fprintf('Statistics --> V0 = %.3f ± %.3f [mm/s], Vd = %.3f ± %.3f [mm/s]\n', v0Mmps.value, (v0Mmps.upper - v0Mmps.lower)/2, vdMmps.value, (vdMmps.upper - vdMmps.lower)/2); % show stat
fprintf('Processing time = %f [sec]\n\n', elapsedTime);

end

function rtrnXYZFrames = convertToMultiexposureFrames(InXYZFrames, CamExposureTime, MultiExposureAlgorithm, FrameRate)
% Read input (video or images) with fixed exposure time and Fps = 1/te and convert it to Multiexposure Laser Speckle images/video frames
% Function parameters:
% InXYZFrames = input frames to process
% CamExposureTime = 250e-6 [sec] etc (cam exposure time in sec.)
% MultiExposureAlgorithm = the increase of exposure time with frame number --> 'lin' (new dt = dt + exposure time) | 'pow2' (new dt = 2*dt)
% FrameRate = desired output frames per second

startTime = tic;

% Start processing
fprintf('\nStart Converting to Multiexposure Laser Speckle Frames... \n'); % show progress

% Start Converting
[rows, cols, frames] = size(InXYZFrames);
rtrnXYZFrames = zeros(rows, cols, frames);
outXYFrame = zeros(rows, cols);
frameTime = 1/FrameRate;

% Recalc frames according algorithm --> note: raw frames are assumed to be taken with the same exposure time and with no delay between frames
switch(MultiExposureAlgorithm)  
    case 'lin' % linear change in exposure time per frame, i.e. dt = dt + CamExposureTime
        i = 0; % output frames counter
        timeStep = 0; % multiexposure time step
        
        for iFrame = 1:frames
            outXYFrame = outXYFrame + InXYZFrames(:, :, iFrame);
            %globalTime = iFrame*CamExposureTime;
            
            % Update multiexposure time step
            timeStep = timeStep + CamExposureTime; % changes in a linear fashion by step = exposure time
            if timeStep > frameTime + 0.1*CamExposureTime % if bigger than frame time + smaller error (avoiding nesty rounding errors)
                % Case the next time step exceed the frame time --> reset
                timeStep = CamExposureTime; % reset multiexposure time
                outXYFrame = InXYZFrames(:, :, iFrame);
            end
            
            i = i + 1;
            rtrnXYZFrames(1:end, 1:end, i) = outXYFrame;
            %outXYZFrames(1:end, 1:end, i) = outXYFrame./max(outXYFrame, [], 'all'); % normalize
        end     
    case 'pow2' % power of 2 change in exposure time per frame, i.e. dt = 2*dt
        i = 0; % output frames counter
        timeStep = CamExposureTime;
        relativeTime = 0;
        
        for iFrame = 1:frames
            outXYFrame = outXYFrame + InXYZFrames(:, :, iFrame); % accumulate frames
            relativeTime = relativeTime + CamExposureTime; % tracks time progression between frames
                                   
            % Update multiexposure time step
            if timeStep > frameTime + 0.1*CamExposureTime % if bigger than frame time + smaller error (avoiding nesty rounding errors)
                % Case the next time step exceed the frame time --> reset
                timeStep = CamExposureTime; % reset multiexposure time
                relativeTime = CamExposureTime;
                outXYFrame = InXYZFrames(:, :, iFrame); % reset frame
            end
            
            % Update pow2 counter time step
            if relativeTime > (timeStep - 0.1*CamExposureTime) && relativeTime < (timeStep + 0.1*CamExposureTime) % if bigger than frame time + smaller error (avoiding nesty rounding errors)
                i = i + 1; % pow2 counter
                rtrnXYZFrames(1:end, 1:end, i) = outXYFrame;
                %outXYZFrames(1:end, 1:end, i) = outXYFrame./max(outXYFrame, [], 'all'); % normalize 
                timeStep = 2*timeStep; % changes in a power of 2 fashion
            end            
        end
        
        % Reduce size of the 3D stack to the actual multiexposure frames
        if frames > 1 && i > 1
            rtrnXYZFrames = rtrnXYZFrames(:, :, 1:i);
        end
    otherwise
        fprintf('\n\nYou have chosen unsupported multiexposure algorithm --> Multiexposure Algorithm = %s\n', MultiExposureAlgorithm);
        error('Exit due to the error above!');
end

% Print progress
elapsedTime = toc(startTime);

fprintf('Multiexposure conversion time = %f [sec]\n', elapsedTime);

end

function rtrnLSPContrast = calcLSPContrastMap(InXYZFrames, CamExposureTime, FrameRate, MultiExposureAlgorithm, MultiExposureGroupFrames)
% Calc the Laser Speckle Contrast Map - return the raw map + processed map (convinient for further processing)

% Calc 3D stack with the spatial LSP contrast K
switch(MultiExposureGroupFrames)
    case 'off' % no grouping of frames for averaging
        InXYZFrames = convertToMultiexposureFrames(InXYZFrames, CamExposureTime, MultiExposureAlgorithm, FrameRate); % convert to multiexposure frames
        rtrnXYZsLSCk = sLASCASumsVectorized1(InXYZFrames); % single LSP Contrast over all frames per multi-exposure
    case 'on' % average LSP Contrast over all grouped frames per multi-exposure
        rtrnXYZsLSCk = sLASCASumsVectorized2(InXYZFrames, CamExposureTime, FrameRate, MultiExposureAlgorithm);
    otherwise
        fprintf('\n\nYou have chosen unsupported multiexposure group frames option --> MultiExposureGroupFrames = %s\n', MultiExposureGroupFrames);
        error('Exit due to the error above!');
end

% Getspatial LSP contrast K into a new structure
[frames] = length(rtrnXYZsLSCk);

% Reshape the calculated LSP 3D stack in more convinient form for further processing - we split differetn multiexposure time series in separate indexes
rtrnLSPContrast = struct([]);
rtrnLSPContrast(1).K = [];
rtrnLSPContrast(1).Kfit = [];
rtrnLSPContrast(1).Time = [];
rtrnLSPContrast(1).globalTime = [];

frameTime = 1/FrameRate;
globalTime = frameTime;
i = 1; % num global time point
timeStep = CamExposureTime; % multiexposure time step
j = 1; % num multiexposure time point (related to tau time)
for iFrame = 1:frames
    % Get current frame of K map in XY for the global time point i and multiexposure time point j
    rtrnLSPContrast(i).K(j) = rtrnXYZsLSCk(iFrame);
    rtrnLSPContrast(i).Time(j) = timeStep; % keep track of the multiexposure time (local time)
    rtrnLSPContrast(i).globalTime = globalTime; % the global time of the i index
    
    % Update multiexposure time step
    newTimeStep = updateTimeStep(MultiExposureAlgorithm, timeStep, CamExposureTime);
    if newTimeStep > frameTime + 0.1*CamExposureTime % if bigger than frame time + smaller error (avoiding nesty rounding errors)
        % Case the next time step exceed the frame time --> reset time step                
        i = i + 1; % update global time counter (we go to the next resolved time point)
        j = 1; % reset multiexposure index counter
        
        timeStep = CamExposureTime; % reset multiexposure time to the initial exposure
        globalTime = i * frameTime; % next global time
    else
        % Case we continue to increase the multiexposure time step since it is less than the frame time        
        timeStep = newTimeStep; % next multieposure time step
        j = j + 1; % update multiexposure index counter
    end
end

% Check and remove if necessary the last element to make sure all global time points 
% have identical size of Time and K arrays (prevents some errors in the Ct(tau) calculation routine)
if length(rtrnLSPContrast) > 1
    if length(rtrnLSPContrast(end).Time) ~= length(rtrnLSPContrast(end-1).Time)
        rtrnLSPContrast(end) = [];
    end
end

end

function rtrnTimeStep = updateTimeStep(MultiExposureAlgorithm, TimeStep, ExposureTime)
% Set/update time step according to chosen algorithm

switch(MultiExposureAlgorithm)
    case 'off'
        rtrnTimeStep = TimeStep + ExposureTime; % fixed time step, but we change it in linear fashon for debugging purposes
    case 'lin'
        rtrnTimeStep = TimeStep + ExposureTime; % changes in a linear fashion by step = exposure time
    case 'pow2'
        rtrnTimeStep = 2*TimeStep; % change as power of 2 of the exposure time
    otherwise
        fprintf('\n\nYou have chosen unsupported multiexposure algorithm --> Multiexposure Algorithm = %s\n', MultiExposureAlgorithm);
        error('Exit due to the error above!');
end
end

function rtrnLSPAutocovariance = calcLSPAutocovarianceMap(LSPContrast)
% Calc the Laser Speckle Autocovaraince Map - calc autocovariance using multiexposure laser speckle contrast

[numGlobalTimePoints] = length(LSPContrast);
[numPointsK] = length(LSPContrast(1).K);

% Calc Autocovariance C(tau)
rtrnLSPAutocovariance = struct([]);
rtrnLSPAutocovariance(1).Ct = [];
rtrnLSPAutocovariance(1).Ctvsfit = [];
rtrnLSPAutocovariance(1).Ctv0vdfit = [];
rtrnLSPAutocovariance(1).Tau = [];
rtrnLSPAutocovariance(1).globalTime = [];

% Set common fit options
fitOptions = fitoptions(...
    'Normalize', 'off',...
    'Exclude', [],...
    'Weights', []);

fprintf('\nProgress, calc Autocovariance Ct(tau) Map: 000.0 [%%] | 00000.0 [sec]');
for i = 1:numGlobalTimePoints
    startTime = tic;
    
    % Get current global time point i and calc Ct(tau) curve for this point
    rtrnLSPAutocovariance(i).globalTime = LSPContrast(i).globalTime; % the global time of the i index
    
    % Precalc - set contrast and time vectors
    T = LSPContrast(i).Time'; % exposure time (also tau) vector    
    numTauPoints = round(T(end)/T(1)); % it will give as many points as possible to fit within frame time by the min exposure time
        
    Tau = linspace(T(1), T(end), numTauPoints); % high number/precision x tau data array ponts
    rtrnLSPAutocovariance(i).Tau = Tau; % keep track of the tau times
    Tau = Tau';
    
    % Calc Ct(tau) by differentiating along tau/time --> 
    % --> Ct(T) = d^2/dT^2 (K^2*T^2)*<I>^2/2, (K - contrast, T - time/exposure-time, <I> - mean intensity, ignored below)    
    sqrT = T.^2; % square of the time
    sqrK = (LSPContrast(i).K .^2)'; % square of the contrast
    
    % Increase differentiation precision by first fitting the curve with a spline (interpolation) function
    %yKT = 0.5 * (K .* T).^2 .* meanIntensity; % y data - function to be differentiated (second derivative with respect to T)
    yKT = 0.5 * (sqrK .* sqrT); % y data - function to be differentiated (second derivative with respect to T)
    
    % Find the second derivative of the function by fine interpolation and using formula for second derivative - it will give the Ct(tau)
    %dTau = Tau(2)-Tau(1);
    %yK = LSPContrast(i).K';
    %yKinterpl = interp1(T, yK, [Tau(1) - dTau; Tau; Tau(end) + dTau], 'spline', 'extrap'); % interpolation func - 'spline' | 'pchip' | etc
    %yKd0 = yKinterpl(2:end-1); % the function as it is or the o derivative of K (which is just K)
    %yKd1 = derivd1(yKinterpl, dTau); % first derivative of K
    %yKd2 = derivd2(yKinterpl, dTau); % second derivative of K
    %CtTau = yKd1.^2 .* Tau.^2 + 4 * yKd0 .* Tau .* yKd1 + yKd0 .* Tau.^2 .* yKd2 + yKd0.^2; % calc Ct(Tau) by expanding the second derivative of T of (0.5 * K(T)^2 * T^2)
    
    % Find the second derivative of the fitted function - it will give the Ct(tau)
    % Fit with 'cubicinterp' (piecewise cubic interpolation) | 'smoothingspline'
    %[fitObj, gofObj, outputObj] = fit(T, yKT, 'smoothingspline', fitOptions); % fit interpolation to make high precision differentiation
    %[~, CtTau] = differentiate(fitObj, Tau); % note: fitObj = fitted ykT function in spline or peicewise polynomial
    
    % Find the second derivative of the function by fine interpolation and using formula for second derivative - it will give the Ct(tau)
    %dTau = Tau(2)-Tau(1);
    %yKTinterpl = interp1(T, yKT, [Tau(1) - dTau; Tau; Tau(end) + dTau], 'spline', 'extrap'); % interpolation func - 'spline' | 'pchip' | etc
    %CtTau = derivd2(yKTinterpl, dTau); % calc the second deriative
    
    % Find the second derivative of the function by fine interpolation and delta operator differantiation - it will give the Ct(tau)
    yKTinterpl = interp1(T, yKT, Tau, 'spline'); % interpolation func - 'spline' | 'pchip' | etc - returns vector of interpl. values
    dTau = Tau(2)-Tau(1);
    CtTau = 4*del2(yKTinterpl, dTau); % calc the second deriative using Laplacian (delta) operator
    
    % Second derivative equals Ct(tau)
    rtrnLSPAutocovariance(i).Ct(1:numTauPoints) = CtTau;
    
    % Fit with 'cubicinterp' (piecewise cubic interpolation) | 'smoothingspline'
%     [fitObj, gofObj, outputObj] = fit(T, yKT, 'smoothingspline', fitOptions); % fit interpolation to make high precision differentiation
%     [~, fitCtTau] = differentiate(fitObj, Tau); % note: fitObj = fitted ykT function in spline or peicewise polynomial
%     
%     figure
%     hold on;
%     lspK = LSPContrast(i).K(:);
%     plot(T, lspK, 'o-', 'DisplayName', 'K');
%     set(gca, 'XScale', 'log'); % set scale
%     legend;
%     hold off;
%     
%     figure
%     hold on;
%     plot(Tau, yKd0, '-', 'DisplayName', 'K(interpl)'); % plot K
%     plot(Tau, yKd1, '-', 'DisplayName', 'dK/dT'); % plot first derivative of K
%     plot(Tau, yKd2, '-', 'DisplayName', 'd2K/dT2'); % plot second derivative of K
%     set(gca, 'XScale', 'log'); % set scale
%     legend('Location','northeast');
%     hold off;
%     
%     figure
%     hold on;
%     yKTfitObj = feval(fitObj, Tau);
%     plot(T, yKT, '.', 'DisplayName', 'yKT(raw)');
%     %plot(Tau, yKTinterpl(2:end-1), '-', 'DisplayName', 'yKT(interpl)');
%     plot(Tau, yKTinterpl, '-', 'DisplayName', 'yKT(interpl)');
%     plot(Tau, yKTfitObj, '-', 'DisplayName', 'yKT(fitObj)');
%     set(gca, 'XScale', 'log'); % set scale
%     legend('Location','northwest');
%     hold off;
%     
%     figure
%     hold on;
%     plot(Tau, CtTau, '-', 'DisplayName', 'CtTau(interpl)');
%     plot(Tau, fitCtTau, '-.', 'DisplayName', 'CtTau(fitObj)');
%     set(gca, 'XScale', 'log'); % set scale
%     legend;
%     hold off;

    elapsedTime = toc(startTime);
    progressPercentage = (i/numGlobalTimePoints)*100;
    remainingTime = (numGlobalTimePoints - i)*elapsedTime;
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'); % delete previous progress line
    fprintf('%05.1f [%%] | %07.1f [sec]', progressPercentage, remainingTime);
end

end

function rtrnLspPSD = calcPsdMap(LSPAutocovariance)
% Calc PSD (Power Spectrum Density) from Autocovariance Map (Ct(tau)) using Wiener-Khinchin theorem =>
% => PSD(freq) = |fft(Intensity(t))|^2 = fft(Ct(tau)) 

[numGlobalTimePoints] = length(LSPAutocovariance);
numPointsTau = length(LSPAutocovariance(1).Ct);

% Calc PSD by FFT of C(tau)
rtrnLspPSD = struct([]);
rtrnLspPSD(1).PsdAmplitude = zeros(numPointsTau);
rtrnLspPSD(1).Freq = [];
rtrnLspPSD(1).globalTime = [];

fprintf('\nProgress, calc PSD(freq) Map: 000.0 [%%] | 00000.0 [sec]');
for i = 1:numGlobalTimePoints
    startTime = tic;
    
    % Get current global time point i and calc PSD for this point
    rtrnLspPSD(i).globalTime = LSPAutocovariance(i).globalTime; % the global time of the i index
    
    % Precalc - set contrast and time vectors    
    Tau = LSPAutocovariance(i).Tau; % high number/precision x tau data array ponts
    %Tau = Tau';
    rtrnLspPSD(i).Freq = flip(1./Tau); % convert tau points in freq (flip to have freq axis from smallest-to-high freq. order)
    
    % Calc PSD(freq) by FFT of Ct(tau)--> we can calc PSD by DCT (direct cosine transform) PSD = 2*DCT(Ct(tau))
    CtTau = LSPAutocovariance(i).Ct(:);
    %CtTau = CtTau(:);
    
    % Use DCT
    dctCt = 2*dct(CtTau); % PSD is the DCT of Ct(tau)
    rtrnLspPSD(i).PsdAmplitude(1:numPointsTau) = dctCt; % PSD amplitude
    
    % Use FFT
    %fftCt = fft(CtTau); % PSD is the FFT of Ct(tau)
    %rtrnLspPSD(i).PsdAmplitude(1:numPointsTau) = real(fftCt); % PSD amplitude
        
    elapsedTime = toc(startTime);
    progressPercentage = (i/numGlobalTimePoints)*100;
    remainingTime = (numGlobalTimePoints - i)*elapsedTime;
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'); % delete previous progress line
    fprintf('%05.1f [%%] | %07.1f [sec]', progressPercentage, remainingTime);        
end

end

function DataLSP = calcVelocityMap(DataLSP, FitMethodForVelocity, WavelengthUm, NA)
% Calc flow velocity by (1) fitting Ct(tau) or (2) fitting Laser Speckle Contrast vs Exposure time

switch(FitMethodForVelocity)
    case 'fit-autocovariance'
        [DataLSP.Velocity, DataLSP.Autocovariance] = calcVelocityMapByAutocovarianceFit(DataLSP.Autocovariance, WavelengthUm, NA);
    case 'fit-lsp-contrast'
        [DataLSP.Velocity, DataLSP.Contrast] = calcVelocityMapByLspContrastFit(DataLSP.Contrast, WavelengthUm, NA);
    otherwise
        fprintf('\n\nUnsupported fit method for velocity --> Velocity Fit Method = %s\n', FitMethodForVelocity);
        error('Exit due to the above error!');
end

end

function [rtrnLspVelocity, LSPAutocovariance] = calcVelocityMapByAutocovarianceFit(LSPAutocovariance, WavelengthUm, NA)
% Calc flow velocity by fitting Ct(tau)

[numGlobalTimePoints] = length(LSPAutocovariance);
numTauPoints = length(LSPAutocovariance(1).Ct);

% The returned velocity structure
rtrnLspVelocity = struct([]);

rtrnLspVelocity(1).vs.value = 0.0; % single velocity map per time point (pure flow model)
rtrnLspVelocity(1).vs.lower = 0.0; % lower confidence interval of vs
rtrnLspVelocity(1).vs.upper = 0.0; % upper confidence interval of vs
rtrnLspVelocity(1).vs.unit = '[mm/s]'; % vs unit

rtrnLspVelocity(1).v0.value = 0.0; % mean velocity map per time point (diffusion + flow model)
rtrnLspVelocity(1).v0.lower = 0.0; % lower confidence interval of v0
rtrnLspVelocity(1).v0.upper = 0.0; % upper confidence interval of v0
rtrnLspVelocity(1).v0.unit = '[mm/s]'; % v0 unit

rtrnLspVelocity(1).vd.value = 0.0; % mean rms velocity (dut to diffusion) map per time point (diffusion + flow model)
rtrnLspVelocity(1).vd.lower = 0.0; % lower confidence interval of vd
rtrnLspVelocity(1).vd.upper = 0.0; % upper confidence interval of vd
rtrnLspVelocity(1).vd.unit = '[mm/s]'; % vd unit

rtrnLspVelocity(1).tc = []; % final tc fitted
rtrnLspVelocity(1).beta = []; % final fitted
rtrnLspVelocity(1).ro = []; % final fitted
rtrnLspVelocity(1).knoise = []; % final fitted

rtrnLspVelocity(1).fitType = 'fit-autocovariance';
rtrnLspVelocity(1).globalTime = [];

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

fprintf('\nProgress, calc Velocity(Time) vs/v0/vd Map (by Ct(tau) fitting): 000.0 [%%] | 00000.0 [sec]');
for i = 1:numGlobalTimePoints
    startTime = tic;
    
    % Get current global time point i and calc velocity vs/v0/vd for this point
    rtrnLspVelocity(i).globalTime = LSPAutocovariance(i).globalTime; % the global time of the i index
    
    % Precalc
    Tau = LSPAutocovariance(i).Tau'; % high number/precision x tau data array ponts
    
    % Calc velocity vs/v0/vd vy fitting of Ct(tau)    
    CtTau = LSPAutocovariance(i).Ct(:);
    
    % Fit Ct(tau) with single flow (vs) model fit function (no difusion or vs >> vd)
    CtTauVsFitFunc = @(vs, x) calcTheoryCtTauVs(x, vs, WavelengthUm, NA); % vs = velosity value to be determined, Tau = x independent variable
    fitOptions.StartPoint = 0; % v --> starting velocity in [um/s]
    fitOptions.Lower = 0; % v --> lower bound of velocity in [um/s]
    [fitObjCtTauVs, gofCtTauVs, outputCtTauVs] = fit(Tau, CtTau, CtTauVsFitFunc, fitOptions); % Tau = x, CtTau = y, CtTauVsFitFunc = singl velocity fit function
    
    vci = confint(fitObjCtTauVs)./1000; % 95% confidence interaval in [mm/s]
    rtrnLspVelocity(i).vs.value = fitObjCtTauVs.vs/1000; % in [mm/s] - directed single flow velocity (no diffusion)
    rtrnLspVelocity(i).vs.lower = vci(1); % in [mm/s]
    rtrnLspVelocity(i).vs.upper = vci(2); % in [mm/s]
    
    % Fit Ct(tau) with mean flow (v0) and mean rms diffusion flow (vd) model fit function
    CtTauV0VdFitFunc = @(v0, vd, x) calcTheoryCtTauV0VdFitFunc(x, v0, vd, WavelengthUm, NA); % v0/vd = velosity value to be determined, Tau = x independent variable
    fitOptions.StartPoint = [fitObjCtTauVs.vs/2, fitObjCtTauVs.vs/2]; % [v0, vd] --> starting velocities in [um/s]
    fitOptions.Lower = [0, 0]; % [v0, vd] --> lower bound of velocities in [um/s]
    [fitObjCtTauV0Vd, gofCtTauV0Vd, outputCtTauV0Vd] = fit(Tau, CtTau, CtTauV0VdFitFunc, fitOptions); % Tau = x, CtTau = y, CtTauV0VdFitFunc = singl velocity fit function
    
    vci = confint(fitObjCtTauV0Vd)./1000; % 95% confidence interaval in [mm/s]
    rtrnLspVelocity(i).v0.value = fitObjCtTauV0Vd.v0/1000; % in [mm/s] - mean directed flow velocity (flow + diffusion model)
    rtrnLspVelocity(i).v0.lower = vci(1, 1); % in [mm/s]
    rtrnLspVelocity(i).v0.upper = vci(2, 1); % in [mm/s]
    
    rtrnLspVelocity(i).vd.value = fitObjCtTauV0Vd.vd/1000; % in [mm/s] - root mean squared velocity due to diffusion (flow + diffusion model)
    rtrnLspVelocity(i).vd.lower = vci(1, 2); % in [mm/s]
    rtrnLspVelocity(i).vd.upper = vci(2, 2); % in [mm/s]
    
    % Calc fitted Ctvsfit and Ctv0vdfit curves
    LSPAutocovariance(i).Ctvsfit = calcTheoryCtTauVs(LSPAutocovariance(i).Tau, fitObjCtTauVs.vs, WavelengthUm, NA);
    LSPAutocovariance(i).Ctv0vdfit = calcTheoryCtTauV0VdFitFunc(LSPAutocovariance(i).Tau, fitObjCtTauV0Vd.v0, fitObjCtTauV0Vd.vd, WavelengthUm, NA);
    
    % Print progress
    elapsedTime = toc(startTime);
    progressPercentage = (i/numGlobalTimePoints)*100;
    remainingTime = (numGlobalTimePoints - i)*elapsedTime;
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'); % delete previous progress line
    fprintf('%05.1f [%%] | %07.1f [sec]', progressPercentage, remainingTime);        
end

end

function [rtrnLspVelocity, LSPContrast] = calcVelocityMapByLspContrastFit(LSPContrast, WavelengthUm, NA)
% Calc flow velocity by fitting K (contrast) vs Time (exposure time) curve (for each pixel)

[numGlobalTimePoints] = length(LSPContrast);
numTauPoints = length(LSPContrast(1).K);

% The returned velocity structure - note: v0 and vd are not used in this fit and are ignored
rtrnLspVelocity = struct([]);

rtrnLspVelocity(1).vs.value = 0.0; % single velocity map per time point (pure flow model)
rtrnLspVelocity(1).vs.lower = 0.0; % lower confidence interval of vs
rtrnLspVelocity(1).vs.upper = 0.0; % upper confidence interval of vs
rtrnLspVelocity(1).vs.unit = '[mm/s]'; % vs unit

rtrnLspVelocity(1).v0 = []; % mean velocity map per time point (diffusion + flow model)
rtrnLspVelocity(1).vd = []; % mean rms velocity (dut to diffusion) map per time point (diffusion + flow model)

rtrnLspVelocity(1).tc.value = 0.0; % final tc fitted value
rtrnLspVelocity(1).tc.lower = 0.0; % lower bound of tc (95% level)
rtrnLspVelocity(1).tc.upper = 0.0; % upper bound of tc (95% level)
rtrnLspVelocity(1).tc.unit = '[s]'; % tc unit

rtrnLspVelocity(1).beta.value = 0.0; % final fitted value
rtrnLspVelocity(1).beta.lower = 0.0; % lower bound (95% level)
rtrnLspVelocity(1).beta.upper = 0.0; % upper bound (95% level)
rtrnLspVelocity(1).beta.unit = '[-]'; % unit

rtrnLspVelocity(1).ro.value = 0.0; % final fitted value
rtrnLspVelocity(1).ro.lower = 0.0; % lower bound (95% level)
rtrnLspVelocity(1).ro.upper = 0.0; % upper bound (95% level)
rtrnLspVelocity(1).ro.unit = '[-]'; % unit

rtrnLspVelocity(1).knoise.value = 0.0; % final fitted value
rtrnLspVelocity(1).knoise.lower = 0.0; % lower bound (95% level)
rtrnLspVelocity(1).knoise.upper = 0.0; % upper bound (95% level)
rtrnLspVelocity(1).knoise.unit = '[-]'; % unit

rtrnLspVelocity(1).fitType = 'fit-lsp-contrast';
rtrnLspVelocity(1).globalTime = [];

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

fprintf('\nProgress, calc Velocity(Time) vs Map (by K(Time) fitting): 000.0 [%%] | 00000.0 [sec]');
for i = 1:numGlobalTimePoints
    startTime = tic;
    
    % Get current global time point i and calc velocity vs/v0/vd for this point
    rtrnLspVelocity(i).globalTime = LSPContrast(i).globalTime; % the global time of the i index
    
    % Precalc
    T = LSPContrast(i).Time'; % exposure time data array ponts
    
    % Calc velocity vs by fitting of K(Time)
    K = LSPContrast(i).K(:);
    
    % Fit K(T) with model fit function too extract the tc (correlation time), P1 (beta = P1^2) and P2 (parameter)
    %KTimeFitFunc = @(tc, P1, P2, x) calcTheoryContrastTimeFitFunc1(x, tc, P1, P2);     
    KTimeFitFunc = @(tc, beta, ro, knoise, x) calcTheoryContrastTimeFitFunc2(x, tc, beta, ro, knoise); % Lorentzian distribution of velocities (single scattering => unordered motion; multiple scattering => ordered motion)
    % KTimeFitFunc = @(tc, beta, ro, knoise, x) calcTheoryContrastTimeFitFunc3(x, tc, beta, ro, knoise); % Gaussian distribution of velocities (ordered motion)
    fitOptions.StartPoint = [1e-6, 0.5, 0.9, 0]; % starting points for the fit parameters --> [tc, beta, ro, knoise]
    fitOptions.Lower = [1e-9, 0, 0, 0]; % lower bound for the fit parameters --> [tc, beta, ro, knoise]
    fitOptions.Upper = [Inf, Inf, 1, Inf]; % upper bound for the fit parameters --> [tc, beta, ro, knoise]
    [fitObjKTime, gofKTime, outputKTime] = fit(T, K, KTimeFitFunc, fitOptions); % T = x, K = y, KTimeFitFunc = contrast fit function
    
    vci = confint(fitObjKTime); % 95% confidence interaval
    
    rtrnLspVelocity(i).tc.value = fitObjKTime.tc; % final tc fitted value
    rtrnLspVelocity(i).tc.lower = vci(1, 1); % lower bound of tc (95% level)
    rtrnLspVelocity(i).tc.upper = vci(2, 1); % upper bound of tc (95% level)
    
    rtrnLspVelocity(i).beta.value = fitObjKTime.beta; % final beta fitted value
    rtrnLspVelocity(i).beta.lower = vci(1, 2); % lower bound of beta (95% level)
    rtrnLspVelocity(i).beta.upper = vci(2, 2); % upper bound of beta (95% level)
    
    rtrnLspVelocity(i).ro.value = fitObjKTime.ro; % final ro fitted value
    rtrnLspVelocity(i).ro.lower = vci(1, 3); % lower bound of ro (95% level)
    rtrnLspVelocity(i).ro.upper = vci(2, 3); % upper bound of ro (95% level)
    
    rtrnLspVelocity(i).knoise.value = fitObjKTime.knoise; % final knoise fitted value
    rtrnLspVelocity(i).knoise.lower = vci(1, 4); % lower bound of knoise (95% level)
    rtrnLspVelocity(i).knoise.upper = vci(2, 4); % upper bound of knoise (95% level)
    
    % Convert correlation time to velocity vs
    rtrnLspVelocity(i).vs.value = calcTheoryTcToVelocity(rtrnLspVelocity(i).tc.value, WavelengthUm, NA)./1000; % in [mm/s] - directed single flow velocity (no diffusion)
    rtrnLspVelocity(i).vs.lower = calcTheoryTcToVelocity(rtrnLspVelocity(i).tc.upper, WavelengthUm, NA)./1000; % in [mm/s]
    rtrnLspVelocity(i).vs.upper = calcTheoryTcToVelocity(rtrnLspVelocity(i).tc.lower, WavelengthUm, NA)./1000; % in [mm/s]
    
    % Calc the respective Kfit (fitted LSP Contrast) using the fitted parameters (i.e. tc, beta, ro, knoise)
    LSPContrast(i).Kfit = calcTheoryContrastTimeFitFunc2(LSPContrast(i).Time, rtrnLspVelocity(i).tc.value, rtrnLspVelocity(i).beta.value, rtrnLspVelocity(i).ro.value, rtrnLspVelocity(i).knoise.value);
    %LSPContrast(i).Kfit = calcTheoryContrastTimeFitFunc3(LSPContrast(i).Time, rtrnLspVelocity(i).tc.value, rtrnLspVelocity(i).beta.value, rtrnLspVelocity(i).ro.value, rtrnLspVelocity(i).knoise.value);
    
    % Print progress
    elapsedTime = toc(startTime);
    progressPercentage = (i/numGlobalTimePoints)*100;
    remainingTime = (numGlobalTimePoints - i)*elapsedTime;
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'); % delete previous progress line
    fprintf('%05.1f [%%] | %07.1f [sec]', progressPercentage, remainingTime);
end

end

function rtrnXYZsLSCk = sLASCASumsVectorized1(InXYZFrames)
% Sums method (vectorized along Z) to calc the Laser Speckle Contrast

[lengthX, lengthY, lengthZ] = size(InXYZFrames);

% Calc 1D contrast map along Z
rtrnXYZsLSCk = zeros(1, lengthZ); % pre-allocate

% Calc Laser Speckle Contrast map --> k = std(I)/<I> = sqrt(<I^2> - <I>^2)/<I> = sqrt(<I^2>/<I>^2 - 1)
fprintf('\nProgress, calc Spatial Laser Speckle Contrast Map: 000.0 [%%] | 00000.0 [sec]');
startTime = tic;

%subFrames = InXYZFrames; % extract subframes XxYxZ given by the window size
meanIntensity = sum(InXYZFrames, [1, 2])./(lengthX*lengthY); % mean intensity along Z
meanSqrIntensity = sum(InXYZFrames .^2, [1, 2])./(lengthX*lengthY); % mean of squared intensities along Z
%meanSqrIntensity = sum(InXYZFrames .^2, [1, 2])./(lengthX*lengthY - 1); % mean of squared intensities along Z
rtrnXYZsLSCk(1:end) = sqrt(meanSqrIntensity - meanIntensity .^2)./meanIntensity; % calc contrast for the given pixel along Z (actually the center pixel of the window)

elapsedTime = toc(startTime);
fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b '); % delete previous progress line
fprintf('%05.1f [%%] | %07.1f [sec]', 100, elapsedTime);
        
        
% Calc with reducing the resolution of the final image by the window size
% Calc Laser Speckle Contrast map --> k = std(I)/<I> = sqrt(<I^2> - <I>^2)/<I> = sqrt(<I^2>/<I>^2 - 1)
% rtrnXYZsLSCk = zeros(ceil(lengthX/XYWindowSizePx), ceil(lengthY/XYWindowSizePx), lengthZ); % pre-allocate
% fprintf('\nProgress, calc Spatial Laser Speckle Contrast Map: 000.0 [%%] | 00000.0 [sec]');
% i = 1;
% j = 1;
% iX = 1;
% iY = 1;
% while iX <= (lengthX - XYWindowSizePx + 1) % loop through image height
%     startTime = tic;
%     while iY <= (lengthY - XYWindowSizePx + 1) % loop through image width
%         subFrames = InXYZFrames(iX:(iX + XYWindowSizePx - 1), iY:(iY + XYWindowSizePx - 1), :); % extract subframes XxYxZ given by the window size
%         meanIntensity = sum(subFrames, [1, 2])./(XYWindowSizePx^2); % mean intensity along Z
%         meanSqrIntensity = sum(subFrames .^2, [1, 2])./(XYWindowSizePx^2); % mean of squared intensities along Z
%         %meanSqrIntensity = sum(subFrames .^2, [1, 2])./(XYWindowSizePx^2 - 1); % mean of squared intensities along Z
%         rtrnXYZsLSCk(i, j, 1:end) = sqrt(meanSqrIntensity - meanIntensity .^2)./meanIntensity; % calc contrast for the given pixel along Z (actually the center pixel of the window)
%         iY = iY + XYWindowSizePx;
%         j = j + 1; % reduced Y pixel counter
%     end
%     elapsedTime = toc(startTime);
%     fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'); % delete previous progress line
%     fprintf('%05.1f [%%] | %07.1f [sec]', (iX/(lengthX - XYWindowSizePx + 1))*100, ((lengthX - XYWindowSizePx + 1)-iX)*elapsedTime);
%     
%     iX = iX + XYWindowSizePx;
%     i = i + 1; % reduced X pixel counter
%     iY = 1; % reset iY
%     j = 1; % reset Y pixel counter
% end

end

function rtrnXYZsLSCk = sLASCASumsVectorized2(InXYZFrames, CamExposureTime, FrameRate, MultiExposureAlgorithm)
% Sums method (vectorized along Z) to calc the Laser Speckle Contrast --> calc mean Contrast using all frames per mulit-exposure time

[lengthX, lengthY, lengthZ] = size(InXYZFrames);

% Calc 1D contrast map along Z
rtrnXYZsLSCk = zeros(1, lengthZ); % pre-allocate

% Calc frame time and number of blocks the 3D stack is split into
frameTime = 1/FrameRate;
numFrames = floor(frameTime/CamExposureTime); % num frames in the current block
if numFrames > lengthZ
    numFrames = lengthZ; % num frames in a block cannot be bigger than the overall avaliable frames
end

% Update multiexposure time step
% timeStep = 0;
% timeStep = timeStep + CamExposureTime; % changes in a linear fashion by step = exposure time
% if timeStep > frameTime + 0.1*CamExposureTime % if bigger than frame time + smaller error (avoiding nesty rounding errors)
%     % Case the next time step exceed the frame time --> reset
%     timeStep = CamExposureTime; % reset multiexposure time
% end

% Update time windows size - represents the type of time step increase
switch(MultiExposureAlgorithm)
    case 'lin'
        %dtWindowPx = dtWindowPx + 1; % represents linear (lin) increase of time step in [px] or frames
        fnUpdateTimeWindow = @(dt)(dt + 1);
    case 'pow2'
        %dtWindowPx = 2*dtWindowPx; % represents exponential (pow2) increase of time step [px] or frames
        fnUpdateTimeWindow = @(dt)(2*dt);
    otherwise
        fprintf('\n\nYou have chosen unsupported multiexposure algorithm --> Multiexposure Algorithm = %s\n', MultiExposureAlgorithm);
        error('Exit due to the error above!');
end

% Calc Laser Speckle Contrast map --> k = std(I)/<I> = sqrt(<I^2> - <I>^2)/<I> = sqrt(<I^2>/<I>^2 - 1)
fprintf('\nProgress, calc Spatial Laser Speckle Contrast Map: 000.0 [%%] | 00000.0 [sec]');

% Calc mean LSP Contrast for each multi-exposure time and each frame block
newLengthZ = numFrames*floor(lengthZ/numFrames); % we cut the frames which cannot fit into the given Z length
jFrame = 0; % global frame counter
for iFrame = 1:numFrames:(newLengthZ - numFrames + 1)
    startTime = tic;
    
    % Extract intensity subframes XxYxZ given by the frames block size
    subFrames = InXYZFrames(:, :, iFrame:(iFrame + numFrames - 1));
    
    % Process the current frames block
    dtWindowPx = 1; % represents time step in [px] or frames
    
    while dtWindowPx <= numFrames
        jFrame = jFrame + 1;
        
        % Reset tmp arrays
        tmpXYZFrames = zeros(lengthX, lengthY);
        %tmpLSPContrast = [];
        
        % Build frames with exposure time given by current time step
        i = 0;        
        for j = 1:dtWindowPx:(numFrames - dtWindowPx + 1)
            i = i + 1;
            tmpXYZFrames(1:end, 1:end, i) = sum(subFrames(:, :, j:(j + dtWindowPx - 1)), 3);
        end
        
        % Variant 1 --> Calc contrast for the respective time step (or multi-exposure time) using the tmp frames
        %meanIntensity = sum(tmpXYZFrames, [1, 2])./(lengthX*lengthY); % mean intensity along Z
        %meanSqrIntensity = sum(tmpXYZFrames .^2, [1, 2])./(lengthX*lengthY); % mean of squared intensities along Z
        %tmpLSPContrast = sqrt(meanSqrIntensity - meanIntensity .^2)./meanIntensity; % calc contrast for the given pixels along XY-Z
        %rtrnXYZsLSCk((iFrame - 1)*numFrames + jFrame) = sum(tmpLSPContrast, 'all')./length(tmpLSPContrast); % calc mean contrast along all frames for the all multi-exposure timee and all frames blocks
        
        % Variant 2 --> Calc contrast for the respective time step (or multi-exposure time) using the tmp frames
        meanIntensity = sum(tmpXYZFrames, 'all')./(lengthX*lengthY*i); % mean intensity along XYZ
        meanSqrIntensity = sum(tmpXYZFrames .^2, 'all')./(lengthX*lengthY*i); % mean of squared intensities along XYZ
        rtrnXYZsLSCk(jFrame) = sqrt(meanSqrIntensity - meanIntensity .^2)./meanIntensity; % calc mean contrast along all frames for the all multi-exposure timee and all frames blocks    
        
        % Update multiexposure time step in terms of window's size in px or frames
        dtWindowPx = fnUpdateTimeWindow(dtWindowPx);
    end
    
    % Print progress
    elapsedTime = toc(startTime);
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b '); % delete previous progress line
    fprintf('%05.1f [%%] | %07.1f [sec]', (iFrame/(newLengthZ - numFrames + 1))*100, ((newLengthZ - numFrames + 1) - iFrame)*elapsedTime);
end

% Reshape LSP array to fit in the global couter frames
rtrnXYZsLSCk = rtrnXYZsLSCk(1:jFrame);

end

function saveLspData(DataLSP, InputFile, PixelXY, XYWindowSizePx, CamExposureTime, FrameRate, WavelengthUm, NA, MultiExposureAlgorithm, MultiExposureGroupFrames, FitMethodForVelocity)
% Save the processed LSP data

% Single XY pixel location to calc/show/save curves
pixY = PixelXY(1);
pixX = PixelXY(2);
YWindowPx = XYWindowSizePx(1);
XWindowPx = XYWindowSizePx(2);

% Options
outputFileType = 'tiff'; % 'tiff'/'tif' | 'avi' | 'mj2'
type3DStackItNormalization = 'global'; % 'global' | 'local'

% Get file name without extension
[inputFilePath, inputFileName, inputFileExtension] = fileparts(InputFile);

% Get file measurement number (if there)
if contains(inputFileName, '.')
    measurementNumber = strtok(inputFileName, '.'); % extract file measurement number
    measurementNumber = [measurementNumber '.'];
else
    measurementNumber = '';
end

% Save LSP Contrast
if ~isempty(DataLSP.Contrast)
    % Build base file name
    %BaseFileName = [measurementNumber 'meLASCA=LSPContrast' sprintf('_t=%fs_me=%s_fps=%d_wl=%.3fum_NA=%f', CamExposureTime, MultiExposureAlgorithm, FrameRate, WavelengthUm, NA)];
    %BaseFileName = [inputFileName sprintf('(me=K(%d,%d)_wl=%.3fum_NA=%.2f)', pixY, pixX, WavelengthUm, NA)];
    BaseFileName = [inputFileName sprintf('_me=%s(gf=%s)_K=(%d,%d)(%dx%d)_wl=%.3fum_NA=%.2f_fps=%.1f',...
        MultiExposureAlgorithm, MultiExposureGroupFrames, pixY, pixX, YWindowPx, XWindowPx, WavelengthUm, NA, FrameRate)];
    
    % Get all (LSP Contrast)
    numPointsK = length(DataLSP.Contrast(1).K);
    globalLength = length(DataLSP.Contrast);
            
    % Save K(Time) curves along Z for given XY (middle) as csv .dat file
    txtFileName = [BaseFileName '.dat'];
    fileId = fopen(txtFileName, 'w'); % open the file for writing
    
    % Check if openning file was successful
    if (fileId == -1)
        error(['Writing to file failed! --> Filepath = ' txtFileName]);  % inform user about the error
    end
    
    % Write header
    for i = 1:globalLength
        fprintf(fileId, 'Time%d[s],K%d(Time)[-]', i, i);
        if ~isempty(DataLSP.Contrast(i).Kfit)
            fprintf(fileId, ',Kfit%d(Time)[-]', i);
        end
        if i < globalLength
            fprintf(fileId, ',');
        end
    end
    
    % Write data    
    for j = 1:numPointsK
        fprintf(fileId, '\n');
        for i = 1:globalLength        
            fprintf(fileId, '%f,%f', DataLSP.Contrast(i).Time(j), DataLSP.Contrast(i).K(j));
            if ~isempty(DataLSP.Contrast(i).Kfit)
                fprintf(fileId, ',%f', DataLSP.Contrast(i).Kfit(j));
            end
            if i < globalLength
                fprintf(fileId, ',');
            end
        end
    end
    
    fclose(fileId);
else
    fprintf('\n\nLSP Contrast data is missing, but is supposed to be present!\n');
    error('Exit due to the error above!');
end

% Save LSP Autocovariance --> save every first measurement in the center of the image for every global time point
if ~isempty(DataLSP.Autocovariance)
    % Build base file name
    %BaseFileName = [measurementNumber 'meLASCA=LSPAutocovariance' sprintf('_t=%fs_me=%s_fps=%d_wl=%.3fum_NA=%f', CamExposureTime, MultiExposureAlgorithm, FrameRate, WavelengthUm, NA)];
    %BaseFileName = [inputFileName sprintf('(me=Ct(%d,%d)_wl=%.3fum_NA=%.2f)', pixY, pixX, WavelengthUm, NA)];
    BaseFileName = [inputFileName sprintf('_me=%s(gf=%s)_Ct=(%d,%d)(%dx%d)_wl=%.3fum_NA=%.2f_fps=%.1f',...
        MultiExposureAlgorithm, MultiExposureGroupFrames, pixY, pixX, YWindowPx, XWindowPx, WavelengthUm, NA, FrameRate)];    
    
    % Get (LSP Autocovariance) curves in the middle of the image
    numPointsCt = length(DataLSP.Autocovariance(1).Ct);
    globalLength = length(DataLSP.Autocovariance);
        
    % Save Ct(tau) curves along Z for given XY (middle) as csv .dat file
    txtFileName = [BaseFileName '.dat'];
    fileId = fopen(txtFileName, 'w'); % open the file for writing
    
    % Check if openning file was successful
    if (fileId == -1)
        error(['Writing to file failed! --> Filepath = ' txtFileName]);  % inform user about the error
    end
    
    % Write header
    for i = 1:globalLength
        fprintf(fileId, 'tau%d[s],Ct%d(tau)[-]', i, i);
        if ~isempty(DataLSP.Autocovariance(i).Ctvsfit)
            fprintf(fileId, ',Ctvsfit%d(tau)[-]', i);
        end
        if ~isempty(DataLSP.Autocovariance(i).Ctv0vdfit)
            fprintf(fileId, ',Ctv0vdfit%d(tau)[-]', i);
        end
        if i < globalLength
            fprintf(fileId, ',');
        end
    end
    
    % Write data    
    for j = 1:numPointsCt
        fprintf(fileId, '\n');
        for i = 1:globalLength        
            fprintf(fileId, '%f,%f', DataLSP.Autocovariance(i).Tau(j), DataLSP.Autocovariance(i).Ct(j));
            if ~isempty(DataLSP.Autocovariance(i).Ctvsfit)
                fprintf(fileId, ',%f', DataLSP.Autocovariance(i).Ctvsfit(j));
            end
            if ~isempty(DataLSP.Autocovariance(i).Ctv0vdfit)
                fprintf(fileId, ',%f', DataLSP.Autocovariance(i).Ctv0vdfit(j));
            end
            if i < globalLength
                fprintf(fileId, ',');
            end
        end
    end
    
    fclose(fileId);
end

% Save LSP PSD (Power Spectrum Density) --> save every first measurement in the center of the image for every global time point
if ~isempty(DataLSP.PSD)
    % Build base file name
    %BaseFileName = [measurementNumber 'meLASCA=LSPPSD' sprintf('_t=%fs_me=%s_fps=%d_wl=%.3fum_NA=%f', CamExposureTime, MultiExposureAlgorithm, FrameRate, WavelengthUm, NA)];
    %BaseFileName = [inputFileName sprintf('(me=PSD(%d, %d)_wl=%.3fum_NA=%.2f)', pixY, pixX, WavelengthUm, NA)];
    BaseFileName = [inputFileName sprintf('_me=%s(gf=%s)_PSD=(%d,%d)(%dx%d)_wl=%.3fum_NA=%.2f_fps=%.1f',...
        MultiExposureAlgorithm, MultiExposureGroupFrames, pixY, pixX, YWindowPx, XWindowPx, WavelengthUm, NA, FrameRate)];
    
    % Get (LSP PSD) curves in the middle of the image
    numPointsPsd = length(DataLSP.PSD(1).PsdAmplitude);
    globalLength = length(DataLSP.PSD);
        
    % Save PSD(freq) curves along Z for given XY (middle) as csv .dat file
    txtFileName = [BaseFileName '.dat'];
    fileId = fopen(txtFileName, 'w'); % open the file for writing
    
    % Check if openning file was successful
    if (fileId == -1)
        error(['Writing to file failed! --> Filepath = ' txtFileName]);  % inform user about the error
    end
    
    % Write header
    for i = 1:globalLength
        fprintf(fileId, 'freq%d[Hz],PSD%d(freq)[-]', i, i);
        if i < globalLength
            fprintf(fileId, ',');
        end
    end
    
    % Write data    
    for j = 1:numPointsPsd
        fprintf(fileId, '\n');
        for i = 1:globalLength        
            fprintf(fileId, '%f,%f', DataLSP.PSD(i).Freq(j), DataLSP.PSD(i).PsdAmplitude(j));
            if i < globalLength
                fprintf(fileId, ',');
            end
        end
    end
    
    fclose(fileId);
end

% Save LSP Contrast
if ~isempty(DataLSP.Velocity)
    % Build base file name
    %BaseFileName = [measurementNumber 'meLASCA=LSPVelocity' sprintf('_t=%fs_me=%s_fps=%d_vfitmethod=%s_wl=%.3fum_NA=%f', CamExposureTime, MultiExposureAlgorithm, FrameRate, FitMethodForVelocity, WavelengthUm, NA)];
    %BaseFileName = [inputFileName sprintf('(me=V(%d,%d)(%s)_wl=%.3fum_NA=%.2f)', pixY, pixX, FitMethodForVelocity, WavelengthUm, NA)];
    BaseFileName = [inputFileName sprintf('_me=%s(gf=%s)_V=(%d,%d)(%dx%d)_wl=%.3fum_NA=%.2f_fps=%.1f',...
        MultiExposureAlgorithm, MultiExposureGroupFrames, pixY, pixX, YWindowPx, XWindowPx, WavelengthUm, NA, FrameRate)];
    
    % Get all (LSP Velocity) frames in one single 3D stack
    %[rows, cols] = size(DataLSP.Velocity(1).vs.value);
    globalLength = length(DataLSP.Velocity);
        
    % Save Velocity(Frame) curves along Z for given XY (middle) as csv .dat file
    txtFileName = [BaseFileName '.dat'];
    fileId = fopen(txtFileName, 'w'); % open the file for writing
    
    % Check if openning file was successful
    if (fileId == -1)
        error(['Writing to file failed! --> Filepath = ' txtFileName]);  % inform user about the error
    end
    
    % Write header
    fprintf(fileId, 'Time[s],vs%s,vslb%s,vsub%s', DataLSP.Velocity(1).vs.unit, DataLSP.Velocity(1).vs.unit, DataLSP.Velocity(1).vs.unit);
    
    if ~isempty(DataLSP.Velocity(i).v0)
        fprintf(fileId, ',v0%s,v0lb%s,v0ub%s', DataLSP.Velocity(1).v0.unit, DataLSP.Velocity(1).v0.unit, DataLSP.Velocity(1).v0.unit);
    end
    
    if ~isempty(DataLSP.Velocity(i).vd)
        fprintf(fileId, ',vd%s,vdlb%s,vdub%s', DataLSP.Velocity(1).vd.unit, DataLSP.Velocity(1).vd.unit, DataLSP.Velocity(1).vd.unit);
    end
    
    if ~isempty(DataLSP.Velocity(i).tc)
        fprintf(fileId, ',tc%s,tclb%s,tcub%s', DataLSP.Velocity(1).tc.unit, DataLSP.Velocity(1).tc.unit, DataLSP.Velocity(1).tc.unit);
    end
    
    if ~isempty(DataLSP.Velocity(i).beta)
        fprintf(fileId, ',beta%s,betalb%s,betaub%s', DataLSP.Velocity(1).beta.unit, DataLSP.Velocity(1).beta.unit, DataLSP.Velocity(1).beta.unit);
    end
    
    if ~isempty(DataLSP.Velocity(i).ro)
        fprintf(fileId, ',ro%s,rolb%s,roub%s', DataLSP.Velocity(1).ro.unit, DataLSP.Velocity(1).ro.unit, DataLSP.Velocity(1).ro.unit);
    end
    
    if ~isempty(DataLSP.Velocity(i).knoise)
        fprintf(fileId, ',knoise%s,knoiselb%s,knoiseub%s', DataLSP.Velocity(1).knoise.unit, DataLSP.Velocity(1).knoise.unit, DataLSP.Velocity(1).knoise.unit);
    end
    
    % Write data    
    for i = 1:globalLength
        fprintf(fileId,'\n'); % go to a new line
        fprintf(fileId, '%f,%f,%f,%f', DataLSP.Velocity(i).globalTime, DataLSP.Velocity(i).vs.value, DataLSP.Velocity(i).vs.lower, DataLSP.Velocity(i).vs.upper);
        
        if ~isempty(DataLSP.Velocity(i).v0)
            fprintf(fileId, ',%f,%f,%f', DataLSP.Velocity(i).v0.value, DataLSP.Velocity(i).v0.lower, DataLSP.Velocity(i).v0.upper);        
        end
        
        if ~isempty(DataLSP.Velocity(i).vd)
            fprintf(fileId, ',%f,%f,%f', DataLSP.Velocity(i).vd.value, DataLSP.Velocity(i).vd.lower, DataLSP.Velocity(i).vd.upper);
        end
        
        if ~isempty(DataLSP.Velocity(i).tc)
            fprintf(fileId, ',%f,%f,%f', DataLSP.Velocity(i).tc.value, DataLSP.Velocity(i).tc.lower, DataLSP.Velocity(i).tc.upper);
        end
        
        if ~isempty(DataLSP.Velocity(i).beta)
            fprintf(fileId, ',%f,%f,%f', DataLSP.Velocity(i).beta.value, DataLSP.Velocity(i).beta.lower, DataLSP.Velocity(i).beta.upper);
        end
        
        if ~isempty(DataLSP.Velocity(i).ro)
            fprintf(fileId, ',%f,%f,%f', DataLSP.Velocity(i).ro.value, DataLSP.Velocity(i).ro.lower, DataLSP.Velocity(i).ro.upper);
        end
        
        if ~isempty(DataLSP.Velocity(i).knoise)
            fprintf(fileId, ',%f,%f,%f', DataLSP.Velocity(i).knoise.value, DataLSP.Velocity(i).knoise.lower, DataLSP.Velocity(i).knoise.upper);
        end
    end
    
    fclose(fileId);
end

end

function f1 = derivd1(f, h)
% Calc numerically derivative d1 (first derivative) - O(h^2) accuracy algorithm for the interiror points
% Formula for the interiror points: f'(x0) = (f(x0 + h) - f(x0 - h))/2*h
% x0 - point where we evaluate the second derivative, h - step size (must be equidistant)
% f - input vector (size must be +2 elements to calc on the border)
% h - step size
% f1 - first derivative of f (size is length(f) - 2, i.e. the border elements are not used)

lengthfVector = length(f);
f1 = zeros(lengthfVector, 1);

if lengthfVector < 5
    fprintf('\n\nNumber of input points is less than 5 --> we need at least 5 points to calc f"\n');
    error('Exit due to the error above!');
end

% Calc the interior points of f'
for i = 2:(lengthfVector - 1)
    f1(i) = (f(i+1) - f(i-1))/(2*h); % calc second derivative in point i
end

% Remove the points on the border
f1 = f1(2:end-1);

end

function f2 = derivd2(f, h)
% Calc numerically derivative d2 (second derivative) - O(h^2) accuracy algorithm for the interiror points
% Formula for the interiror points: f''(x0) = (f(x0 + h) - 2*f(x0) + f(x0 - h))/h^2
% x0 - point where we evaluate the second derivative, h - step size (must be equidistant)
% f - input vector (size must be +2 elements to calc on the border)
% h - step size
% f2 - second derivative of f (size is length(f) - 2, i.e. the border elements are not used)

lengthfVector = length(f);
f2 = zeros(lengthfVector, 1);

if lengthfVector < 5
    fprintf('\n\nNumber of input points is less than 5 --> we need at least 5 points to calc f"\n');
    error('Exit due to the error above!');
end

% Calc the interior points of f"
for i = 2:(lengthfVector - 1)
    f2(i) = (f(i+1) - 2*f(i) + f(i-1))/h^2; % calc second derivative in point i
end

% Remove the points on the border
f2 = f2(2:end-1);

end

function rtrnCtTauZ = calcMeanCtTauZ(CtTauXYZ)
% Calculate mean Ct(tau) from neiboughour XY-Z (i.e. along Z)

[lengthX, lengthY, lengthZ] = size(CtTauXYZ);

% Calc mean Ct(tau)
rtrnCtTauZ = sum(CtTauXYZ, [1, 2])./(lengthX*lengthY);

end

function rtrnCt = calcTheoryCtTauVs(Tau, Vs, Wavelength, NA)
% Calculate Ct(tau, v) with no diffusion (i.e. velocity >> diffusion), Ct(tau, vs) = exp(-(M*v*tau)^2/len0^2), l0 = decorelation length (see eq. 9 in the paepr below)
% Hao Li et al, "Directly measuring absolute flow speed by frequency-domain laser speckle imaging", DOI link: https://doi.org/10.1364/OE.22.021079
% Vs = single/direted flow velocity [Wavelength]/s (unit depends on the wavelength unit)
% Wavelength = illumintation wavelength (its unit will define the velocity unit)
% NA = numerical aperture
% Note: Magnification M cancels out

% Calc decorelation length
len0 = 0.41*Wavelength/NA;

% Calc Ct(tau, v) --> Gaussian function
rtrnCt = exp(-(Vs.*Tau).^2./len0^2);

end

function rtrnCt = calcTheoryCtTauV0VdFitFunc(Tau, V0, Vd, Wavelength, NA)
% Calculate Ct(tau, v) with diffusion, i.e. Ct(tau, v0, vd), len0 = decorelation length (see eq. 13 in the paper below)
% Hao Li et al, "Directly measuring absolute flow speed by frequency-domain laser speckle imaging", DOI link: https://doi.org/10.1364/OE.22.021079
% V0 = mean velocity (directed velocity due to flow) [Wavelength]/s (unit depends on the wavelength unit)
% Vd = root mean square velocity (due to diffusion) [Wavelength]/s (unit depends on the wavelength unit)
% Wavelength = illumintation wavelength (its unit will define the velocity unit)
% NA = numerical aperture
% Note: Magnification M cancels out

% Calc decorelation length
len0 = 0.41*Wavelength/NA;

% Precalc
expr1 = len0^2 + (Vd.*Tau).^2;

% Calc Ct(tau, v0, vd)
Ct1 = exp(-(V0.*Tau).^2./expr1);
Ct2 = len0^3./expr1.^1.5 + (2*V0.^2.*Vd.^2.*len0.*Tau.^4)./expr1.^2.5;

rtrnCt = Ct1.*Ct2;

end

function rtrnK = calcTheoryContrastTimeFitFunc1(Time, Tc, P1, P2)
% Calcs contrast K as function of Time (exposure time), Tc (correlation/decorelation time) and P1 (a parameter, note: beta = P1^2), and P2 (parameter)
% Note: this fit function accounts for the static speckle, e.g. due to semi-transparent scattering layer(s)
% Equation is based on eq. 6 in the following paper:
% Tomi Smausz et al, "Real correlation time measurement in laser speckle contrast analysis using wide exposure time range images", DOI:https://doi.org/10.1364/AO.48.001425
% Time = exposure time
% Tc = correlation time (where Ct(tau) decays to 1/e)
% P1 = parameter (beta = P^2) to account for static speckle and pixel/speckle size
% P2 = parameter to account for the static speckle

% Precalc
x = Time./Tc;

% Calc contrast - based on eq. 6 in the paper above
rtrnK = P1 * sqrt((1./(2*x.^2)) .* (exp(-2*x) - 1 + 2*x) + P2);

end

function rtrnK = calcTheoryContrastTimeFitFunc2(Time, tc, beta, ro, knoise)
% Calcs (Lorentizian distribution) contrast K as function of Time (exposure time), tc ((de)corelation time) and beta, ro, knoise
% Lorentzian distribution of velocities (single scattering => unordered motion; multiple scattering => ordered motion)
% Note: this fit function accounts for the static speckle (e.g. due to semi-transparent scattering layer(s)) + noise (camera and/or experimental artifacts)
% Equation is based on eq. 11 in the following paper:
% Ashwin B. Parthasarathy et al, "Robust flow measurement with multi-exposure speckle imaging", DOI:https://doi.org/10.1364/OE.16.001975
% Equation is based on eq. 10 in the following paper:
% Ping Kong et al, "A novel highly efficient algorithm for laser speckle imaging", DOI:https://doi.org/10.1016/j.ijleo.2016.04.004
% Equation is based on eq. 3 in the following paper:
% S. Kazmi et al, "Evaluating multi-exposure speckle imaging estimates of absolute autocorrelation times", DOI:https://doi.org/10.1364/OL.40.003643
% Equation is based on eq. 14 in the following paper:
% D. Boas et al, "Laser speckle contrast imaging in biomedical optics", DOI:https://doi.org/10.1117/1.3285504
% Time = exposure time
% tc = correlation time (where Ct(tau) decays to 1/e)
% beta = normalization factor to account for speckle averaging due to mismatch of speckle size and detector size, polarization and coherence effects
% ro = fraction of the total light that is dynamically scattered (between 0 and 1)
% knoise = constant variance due to experimental background noise (shot noise + camera noise)

% Precalc
x = Time./tc;

% Calc contrast (temporal) - based on eq. 10 and/or 11 in the papers above
%rtrnK = sqrt(beta .* ro.^2 .*(exp(-2*x) + 2*x - 1)./(2*x.^2) + 4 .* beta .* ro .* (1 - ro) .* (exp(-x) + x - 1)./x.^2 + knoise);

% % Calc contrast (spatial) - based on eq. 3 in the paper above
%rtrnK = sqrt(beta .* ro.^2 .*(exp(-2*x) + 2*x - 1)./(2*x.^2) + 4 .* beta .* ro .* (1 - ro) .* (exp(-x) + x - 1)./x.^2 + beta .* (1 - ro).^2 + knoise);

% % Calc contrast (spatial) - based on eq. 14 in the paper above
rtrnK = sqrt(beta .* ro.^2 .*(exp(-2*x) + 2*x - 1)./(2*x.^2) + 4 .* beta .* ro .* (1 - ro) .* (exp(-x) + x - 1)./x.^2 + beta .* (1 - ro).^2) + knoise;

end

function rtrnK = calcTheoryContrastTimeFitFunc3(Time, tc, beta, ro, knoise)
% Calcs (Gaussian distribution) contrast K as function of Time (exposure time), tc ((de)corelation time) and beta, ro, knoise
% Gaussian distribution of velocities (single scattering => ordered motion)
% Note: this fit function accounts for the static speckle (e.g. due to semi-transparent scattering layer(s)) + noise (camera and/or experimental artifacts)
% Equation is based on eq. 11 in the following paper:
% Mitchell A. Davis et al, "Sensitivity of laser speckle contrast imaging to flow perturbations in the cortex", DOI:https://doi.org/10.1364/BOE.7.000759 
% Equation is based on eq. 2 in the following paper:
% Ashwin B. Parthasarathy et al, "Quantitative imaging of ischemic stroke through thinned skull in mice with Multi Exposure Speckle Imaging", DOI:https://doi.org/10.1364/BOE.1.000246
% Equation is based on eq. 7 in the following paper:
% Annemarie Nadort et al, "Quantitative laser speckle flowmetry of the in vivo microcirculation using sidestream dark field microscopy", DOI:https://doi.org/10.1364/BOE.4.002347
% Time = exposure time
% tc = correlation time (where Ct(tau) decays to 1/e)
% beta = normalization factor to account for speckle averaging due to mismatch of speckle size and detector size, polarization and coherence effects
% ro = fraction of the total light that is dynamically scattered (between 0 and 1)
% knoise = constant variance due to experimental background noise (shot noise + camera noise)

% Precalc
x = Time./tc;

% Calc contrast - based on eq. 11 in the paper above
%rtrnK = sqrt(beta.*(sqrt(2*pi)*erf(sqrt(2)*x))./(2*x) - beta.*(1 - exp(-2*x.^2))./(2*x.^2));

% % Calc contrast (spatial) - based on eq. 7 (modified eq. 2) in the paper above
rtrnK = sqrt(beta .* ro.^2 .*(exp(-2*x.^2) - 1 + (sqrt(2*pi).*x.*erf(sqrt(2)*x)))./(2*x.^2) + 2 .* beta .* ro .* (1 - ro) .* (exp(-x.^2) - 1 + (sqrt(pi.*x).*erf(x)))./(x.^2) + beta .* (1 - ro).^2) + knoise;

end

function rtrnVs = calcTheoryTcToVelocity(Tc, Wavelength, NA)
% Calculate single velocity from Tc (correlation time) and len0 (correlation length)
% Tc = correlation/decorrelation time (where Ct(tau) = 1/e)
% Wavelength = wavelength of the illumination
% NA = numerical aperture of the optical system
% Note: the Vs (velocity) units will depend on Tc unit and Wavelength unit, e.g. if Tc in [s] and Wavelegnth in [um] => Vs in [um/s]

% Calc decorrelation length
len0 = 0.41*Wavelength/NA;

% Calc velocity Vs --> we use the formula Ct(Tau) = exp(-(Vs*Tau)^2/len0^2) = exp(- Tau^2/Tc^2) => tc = len0/Vs => Vs = len0/Tc
rtrnVs = len0./Tc;

end
