function lsci_fftLSA(InputFile, NumericalMethod, StartFrame, EndFrame, ZWindowSizeFrames, CamExposureTime, FrameRate, PixelXY, PixelWindowXY, CamPixelSizeUm, Magnification, WavelengthUm, NA)
% Read file, analyse laser speckle by FFT and frequency domain laser speckle (1D Temproal and 2D Spatial)
% InputFile = file name of the input data (if empty brings command line file dialog) - supports avi (video) | mj2 (video Motion Jpeg 2000) | tiff (multipage)
% NumericalMethod = calc Ct(tau) using 'fft' (Fast Fourier Transform) or 'xcov' (built in matlab (auto)-covariance function)
% Process between StartFrame and EndFrame frames.
% ZWindowSizeFrames = 500, 1000 etc (pixel size of the Z sliding window to calc Ct(tau) - the autocovariance)
% CamExposureTime = 250e-6 [sec] etc (cam exposure time in sec.)
% FrameRate = frames per second
% PixelXY = [X, Y] (coordinate of the point where we calc FFT, Ct(tau) etc)
% PixelWindowXY = [X, Y] size in pixels over which we average the Ct(tau)
% CamPixelSizeUm = physical sze of a pixel in [um]
% Magnification = magnification of the optical system
% WavelengthUm = wavelength of illumination in [um]
% NA = numerical aperture of the imaging system

PixelSize = CamPixelSizeUm/Magnification; % pixel size (it will depend on the magnification)

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

% Write frames to multipage tiff file
fprintf('\nStart calculating Temporal Frequency Domain Laser Speckle... \n'); % show progress

% Process input
pixX = PixelXY(2); % vertical coordinate wehre we calc FDLS Ct(tau) curve(s)
pixY = PixelXY(1); % horizontal coordinate wehre we calc FDLS Ct(tau) curve(s)
signal1Z = inXYZFrames(pixX, pixY, :);
signal1Z = signal1Z(:);
%signal1Z = signal1Z - mean(signal1Z, 'all'); % remove the mean of the signal
fftSignal1Z = fft(signal1Z); % FFT of the center along Z
%fft2InputVideo = fft2(inXYZFrames); % FFT of each frame

%fft2AbsInputVideo = abs(fft2InputVideo);
fftAbsSignal1Z = abs(fftSignal1Z); % power spectrum
fftPSDSignal1Z = fftAbsSignal1Z.^2; % power spectrum density (PSD)
 
% Calc autocovariance
halfPixelWindowX = floor(PixelWindowXY(2)/2);
halfPixelWindowY = floor(PixelWindowXY(1)/2);
signal2Z = inXYZFrames((pixX - halfPixelWindowX + 1):(pixX + halfPixelWindowX), (pixY - halfPixelWindowY + 1):(pixY + halfPixelWindowY), :); % get XY-Z 3D slice
CtTauXYZ = tFDLSIDirect(signal2Z, ZWindowSizeFrames, NumericalMethod); % calc by FFT or xcov
Cs2DXYZ = sFDLSIDirect(signal2Z, ZWindowSizeFrames);
CstTauXYZ = stFDLSIDirect(inXYZFrames, ZWindowSizeFrames, NumericalMethod, pixX, pixY, pixX + 0, pixY + 20); % cross-correlation between the given pixels

%signal2Z = permute(signal2Z, [3, 1, 2]);
%signal2Z = signal2Z - mean(signal2Z, 'all'); % remove the mean of the signal
fftSignal2Z = fft(signal2Z, [], 3); % calc all FFT along Z
fftSignal2Z = mean(fftSignal2Z, [1, 2]); % average power spectrum
fftSignal2Z = fftSignal2Z(:);
fftAbsSignal2Z = abs(fftSignal2Z); % average power spectrum
fftPSDSignal2Z = fftAbsSignal2Z.^2; % average power spectrum density (PSD)

CtTauXYZ = mean(CtTauXYZ, [1, 2]);
CtTauXYZ = CtTauXYZ(:);

CstTauXYZ = permute(CstTauXYZ, [2, 1]);

% Save result
tiffFileNamePathFDLSfft2 = fullfile(filePath, [fileName '_tFDLS-2D-XY' '.tiff']); % Assemble tiff file name
tiffFileNamePathFDLSfft = fullfile(filePath, [fileName '_tFDLS-1D-Z' '.tiff']); % Assemble tiff file name
% for i = 1:size(fft2AbsInputVideo, 3)
%     %cInt8 = 255; % coefficient to convert to 8 bit integer
%     %frameK = uint8(cInt8*outXYZtFDLStc(:, :, i)); % convert to 16 bit depth
%     %frameV = uint8(cInt8*outXYZtFDLSv(:, :, i)); % convert to 16 bit depth
%     
%     cInt16 = 2^16 - 1; % coefficient to convert to 16 bit integer
%     fft2Frame = log10(fft2AbsInputVideo(:, :, i));
%     fft2Frame = fft2Frame./max(fft2Frame, [], 'all'); % convert to 16 bit depth
%     fft2Frame = uint16(cInt16*fft2Frame); % convert to 16 bit depth
%     fft2Frame = fftshift(fft2Frame);
%     
%     imwrite(fft2Frame, tiffFileNamePathFDLSfft2, 'tiff', 'Compression', 'packbits', 'WriteMode', 'append'); % save
% end

% Prepare X scales
pixNumX = (0:length(signal1Z)-1)';
pixTimeX = (pixNumX)./FrameRate; % calc time scale
pixFreqX = ((0:length(fftAbsSignal1Z)-1)*FrameRate/length(fftAbsSignal1Z))'; % calc frequency scale

pixSpecialX = (1:length(Cs2DXYZ))' .* PixelSize; % calc spatial scale

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
    'Display', 'notify',...
    'MaxFunEvals', 20000,...
    'MaxIter', 5000,...
    'TolFun', 1.0000e-09,...
    'TolX', 1.0000e-09);

% Fit temporal autocovariance with single flow velocity function (no difusion)
CtV0Tau = @(Velocity, x) calcTheoryCtVTau(x, Velocity, WavelengthUm, NA); % Velocity = coefficient to be determined, Tau = x independent variable
fitOptions.StartPoint = 100; % v --> starting velocity in [um/s]
fitOptions.Lower = 0; % v --> lower bound of velocity in [um/s]
[fitObjectCtV0Tau, gofCtV0Tau, outputCtV0Tau] = fit(pixTimeX, CtTauXYZ, CtV0Tau, fitOptions); % pixTimeX = x, CtTauXYZ = y, CtV0Tau = singl velocity fit function
vci = confint(fitObjectCtV0Tau)./1000; % 95% confidence interaval in [mm/s]
vMmps.value = fitObjectCtV0Tau.Velocity/1000; % [mm/s] - directed single flow velocity (no diffusion)
vMmps.lower = vci(1);
vMmps.upper = vci(2);

% Fit temporal autocovariance with mean flow velocity and diffusion
CtV0VdTau = @(V0, Vd, x) calcTheoryCtV0VdTau(x, V0, Vd, WavelengthUm, NA); % Velocity = coefficient to be determined, Tau = x independent variable
fitOptions.StartPoint = [fitObjectCtV0Tau.Velocity/2, fitObjectCtV0Tau.Velocity/2]; % [v0, vd] --> starting velocities in [um/s]
fitOptions.Lower = [0, 0]; % [v0, vd] --> lower bound of velocities in [um/s]
[fitObjectCtV0VdTau, gofCtV0VdTau, outputCtV0VdTau] = fit(pixTimeX, CtTauXYZ, CtV0VdTau, fitOptions); % pixTimeX = x, CtTauXYZ = y, CtV0Tau = singl velocity fit function
v0vdci = confint(fitObjectCtV0VdTau)./1000; % 95% confidence interaval in [mm/s]
v0Mmps.value = fitObjectCtV0VdTau.V0/1000; % [mm/s] - directed single flow mean velocity (with diffusion)
v0Mmps.lower = v0vdci(1, 1);
v0Mmps.upper = v0vdci(2, 1);
vdMmps.value = fitObjectCtV0VdTau.Vd/1000; % [mm/s] - root mean squared velocity (due to diffusion)
vdMmps.lower = v0vdci(1, 2);
vdMmps.upper = v0vdci(2, 2);

% Plot results
figure;
plot(pixTimeX, signal1Z, '-', 'Color', 'blue'); % plot input signal
set(gca, 'FontSize', 10); % set font size
set(gca, 'YScale', 'linear'); % set scale
set(gca, 'Box', 'on'); % set plot to a box
title(gca, 'Signal Z(Temporal)');
xlabel(gca, 'Time [s]');
ylabel(gca, 'Intensity(a.u.)');

figure;
hold on;
plot(pixFreqX(2:round(length(pixFreqX)/2)), fftAbsSignal1Z(2:round(length(pixFreqX)/2)), '-', 'DisplayName', 'single', 'Color', 'red'); % plot FFT magnitude of input signal
plot(pixFreqX(2:round(length(pixFreqX)/2)), fftAbsSignal2Z(2:round(length(pixFreqX)/2)), '-', 'DisplayName', 'mean', 'Color', 'blue'); % plot FFT magnitude of input signal (averaged over XY window)
set(gca, 'FontSize', 10); % set font size
set(gca, 'YScale', 'log'); % set scale
set(gca, 'Box', 'on'); % set plot to a box
title(gca, 'Signal Spectrum Z (Temporal)');
xlabel(gca, 'Frequency [Hz]');
ylabel(gca, 'Power(a.u.)');
legend;
hold off;

figure;
hold on;
plot(pixFreqX(2:round(length(pixFreqX)/2)), fftPSDSignal1Z(2:round(length(pixFreqX)/2)), '-', 'DisplayName', 'single', 'Color', 'green'); % plot FFT Power Spectrum Dnesity of input signal
plot(pixFreqX(2:round(length(pixFreqX)/2)), fftPSDSignal2Z(2:round(length(pixFreqX)/2)), '-', 'DisplayName', 'mean', 'Color', 'black'); % plot FFT Power Spectrum Dnesity of input signal
set(gca, 'FontSize', 10); % set font size
set(gca, 'YScale', 'log'); % set scale
set(gca, 'Box', 'on'); % set plot to a box
title(gca, 'Signal Power Spectrum Density Z (Temporal)');
xlabel(gca, 'Frequency [Hz]');
ylabel(gca, 'PSD(a.u.)');
legend;
hold off;

figure;
hold on;
plot(pixTimeX+pixTimeX(2), CtTauXYZ, '.', 'DisplayName', 'experiment', 'Color', [1 0 0]); % plot Ct magnitude of input signal (temporal)
plot(pixTimeX+pixTimeX(2), fitObjectCtV0Tau(pixTimeX), '-', 'DisplayName', ['fit (single flow): v=' num2str(vMmps.value, 4) '[mm/s]'], 'Color', [0 0 0]); % plot fit of input signal (temporal)
plot(pixTimeX+pixTimeX(2), fitObjectCtV0VdTau(pixTimeX), '--', 'DisplayName', ['fit (flow + diffusion): v0=' num2str(v0Mmps.value, 4) '[mm/s], vd=' num2str(vdMmps.value, 4) '[mm/s]'], 'Color', [0 0.5 1]); % plot fit of input signal (temporal)
set(gca, 'FontSize', 10); % set font size
set(gca, 'XScale', 'log'); % set scale
set(gca, 'Box', 'on'); % set plot to a box
title(gca, 'Autocovariance');
xlabel(gca, 'tau [s]');
ylabel(gca, 'Ct(tau) (Normalized)');
%xlim([0, 5*pixTimeX(2)]);
ylim([-0.5, 1.1]);
legend('location', 'southwest');
hold off;

% figure;
% plot(pixSpecialX, Cs2DXYZ, '-', 'DisplayName', 'Cs(x,y)', 'Color', [0 0 1]); % plot FFT magnitude of input signal (spatial)
% set(gca, 'FontSize', 10); % set font size
% set(gca, 'XScale', 'log'); % set scale
% set(gca, 'Box', 'on'); % set plot to a box
% title(gca, 'Autocovariance');
% xlabel(gca, 'spatial [um]');
% ylabel(gca, 'Cs(x,y) (Normalized)');
% legend;

figure;
hold on;
plot(pixTimeX+pixTimeX(2), CstTauXYZ, '-', 'DisplayName', 'experiment', 'Color', [1 0 1]); % plot cross-covariance
set(gca, 'FontSize', 10); % set font size
set(gca, 'XScale', 'log'); % set scale
set(gca, 'Box', 'on'); % set plot to a box
title(gca, 'Crosscovariance');
xlabel(gca, 'tau [s]');
ylabel(gca, 'Cst(tau)');
%xlim([0, 5*pixTimeX(2)]);
%ylim([-0.1, 1]);
legend;
hold off;

% Save data
DataLSP.CtTauXYZ.Ctex = CtTauXYZ; % experimental auto-covariance
DataLSP.CtTauXYZ.CtfitV0 = fitObjectCtV0Tau(pixTimeX); % fitted auto-covariance (single flow model)
DataLSP.CtTauXYZ.CtfitV0Vd = fitObjectCtV0VdTau(pixTimeX); % fitted auto-covariance (flow + diffusion model)
DataLSP.CtTauXYZ.FitParms.SingleFlowModel.V0 = vMmps;
DataLSP.CtTauXYZ.FitParms.FlowDiffModel.V0 = v0Mmps;
DataLSP.CtTauXYZ.FitParms.FlowDiffModel.Vd = vdMmps;
DataLSP.CtTauXYZ.Tau = pixTimeX+pixTimeX(2);
DataLSP.PSD.Amplitude = fftPSDSignal2Z(2:round(length(pixFreqX)/2));
DataLSP.PSD.Freq = pixFreqX(2:round(length(pixFreqX)/2));
saveLspData(DataLSP, InputFile, PixelXY, PixelWindowXY, CamExposureTime, FrameRate, WavelengthUm, NA);

% Print progress
elapsedTime = toc(startTime);

fprintf('\n\nEnd of processing --> Start Frame = %d, End Frame = %d\n', StartFrame, EndFrame); % show progress
fprintf('Statistics --> Max Freq Ampl = %f, Mean Freq Ampl = %f\n', max(fftAbsSignal1Z), mean(fftAbsSignal1Z)); % show stat
fprintf('Statistics Pixel[%d, %d] --> V (single flow) = %.3f ± %.3f [mm/s]\n', pixY, pixX, vMmps.value, (vMmps.upper - vMmps.lower)/2); % show stat
fprintf('Statistics Pixel[%d, %d] --> V0 = %.3f ± %.3f [mm/s], Vd = %.3f ± %.3f [mm/s]\n', pixY, pixX, v0Mmps.value, (v0Mmps.upper - v0Mmps.lower)/2, vdMmps.value, (vdMmps.upper - vdMmps.lower)/2); % show stat
fprintf('Processing time = %f [sec]\n\n', elapsedTime);

end

function rtrnCtTauXYZ = tFDLSIDirect(InXYZFrames, ZWindowSizeFrames, NumericalMethod)
% Calculate autocovariance (for each pixel in XY) of intensity signal by FFT or xcov() methods

% Calc Ct(tau), i.e. the auto-covariance
switch(NumericalMethod)
    case 'fft' % Calc using FFT
        rtrnCtTauXYZ = tAutoCovarianceFFT(InXYZFrames, ZWindowSizeFrames);        
    case 'xcov' % calc using buit in matalb covariance xcov() function
        rtrnCtTauXYZ = tAutoCovarianceXcov(InXYZFrames, ZWindowSizeFrames);
    otherwise
        fprintf('\n\nUnsupported numerical method: %s\', NumericalMethod);
        error('Exit due to the above error!');
end

end

function rtrnCtTauXYZ = tAutoCovarianceFFT(InXYZFrames, ZWindowSizeFrames)
% Calculate autocovariance (for each pixel in XY) of intensity signal by Fourier Transform (see eq 1, 3 and 4 in the paper below)
% Hao Li et al, "Directly measuring absolute flow speed by frequency-domain laser speckle imaging", DOI link: https://doi.org/10.1364/OE.22.021079

[lengthX, lengthY, lengthZ] = size(InXYZFrames);
fitLengthZ = floor(lengthZ/ZWindowSizeFrames); % how many times the z window fits in all frames

% Pre-allocate
rtrnCtTauXYZ = zeros(lengthX, lengthY, ZWindowSizeFrames*fitLengthZ); % it can contain more than one Ct(tau9 curve as many as can be fitted in the size

% Calc 
fprintf('\nProgress, calc Ct(tau) by FFT: 000.0 [%%] | 00000.0 [sec]');
iZBegin = 1;
for iZ = 1:fitLengthZ % loop as many times as ZWindowSizeFrames fits in all available frames
    startTime = tic;
    % Calc autocovariance Ct(tau) = <I°(t)I(t+tau)> - <I>^2
    for iX = 1:lengthX % loop through image height
        for iY = 1:lengthY % loop through image width
            subFrame = InXYZFrames(iX, iY, iZBegin:(iZBegin+ZWindowSizeFrames-1)); % extract subframe given by the window size
            subFrame = permute(subFrame, [3, 1, 2]);
            meanIntensity = sum(subFrame, 'all')/ZWindowSizeFrames; % mean intensity            
            subFrameZeroPadded = [subFrame - meanIntensity; zeros(ZWindowSizeFrames, 1)]; % add zero padding to calc correctly the autocovariance by FFT
            
            Iw = fft(subFrameZeroPadded); % calc the Fourier Transform of the input intensity along temporal direction I(w) = F(I(t)) = 1/2Pi*integral(I(t)*exp(-I*w*t)*dt)
            sqrIw = abs(Iw).^2; % the square amplitude of the power spectrum of the input signal
            CtTau = ifft(sqrIw); % calc Ct(tau) as FT of the square of the power spectrum of the input signal (plus remove the contribution of the DC term)
            
            CtTau = CtTau(1:ZWindowSizeFrames); % skips out the zero padding part
            CtTauNormalized = CtTau./CtTau(1); % normalize to remove the dependence on the illumination intensity
            rtrnCtTauXYZ(iX, iY, iZBegin:(iZBegin+ZWindowSizeFrames-1)) = CtTauNormalized; % the normalized autocovariance for the given pixel XY and window along Z
        end
    end
    iZBegin = iZ*ZWindowSizeFrames;
    elapsedTime = toc(startTime);
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'); % delete previous progress line
    fprintf('%05.1f [%%] | %07.1f [sec]', (iZ/fitLengthZ)*100, (fitLengthZ-iZ)*elapsedTime);
end

end

function rtrnCtTauXYZ = tAutoCovarianceXcov(InXYZFrames, ZWindowSizeFrames)
% Calculate autocovariance (for each pixel in XY) of intensity signal by matlab xcov

[lengthX, lengthY, lengthZ] = size(InXYZFrames);
fitLengthZ = floor(lengthZ/ZWindowSizeFrames); % how many times the z window fits in all frames

% Pre-allocate
rtrnCtTauXYZ = zeros(lengthX, lengthY, ZWindowSizeFrames*fitLengthZ); % it can contain more than one Ct(tau) curve as many as can be fitted in the size

% Calc 
fprintf('\nProgress, calc Ct(tau) by xcov(): 000.0 [%%] | 00000.0 [sec]');
iZBegin = 1;
for iZ = 1:fitLengthZ % loop as many times as ZWindowSizeFrames fits in all available frames
    startTime = tic;    
    % Calc autocovariance Ct(tau) = xcov(I(t))
    for iX = 1:lengthX % loop through image height
        for iY = 1:lengthY % loop through image width
            subFrame = InXYZFrames(iX, iY, iZBegin:(iZBegin+ZWindowSizeFrames-1)); % extract subframe given by the window size
            CtTau = xcov(subFrame(:)); % return the autocovariance
            CtTauNormalized = CtTau./max(CtTau); % normalize to remove the dependence on the illumination intensity
            rtrnCtTauXYZ(iX, iY, iZBegin:(iZBegin+ZWindowSizeFrames-1)) = CtTauNormalized(ZWindowSizeFrames:end); % the normalized autocovariance for the given pixel XY and window along Z
        end
    end
    iZBegin = iZ*ZWindowSizeFrames;
    elapsedTime = toc(startTime);
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'); % delete previous progress line
    fprintf('%05.1f [%%] | %07.1f [sec]', (iZ/fitLengthZ)*100, (fitLengthZ-iZ)*elapsedTime);
end

end

function rtrnCsTauXYZ = sFDLSIDirect(InXYZFrames, ZWindowSizeFrames)
% Calculate spatial autocovariance (for window in XY) of intensity signal by xcov()

[lengthX, lengthY, lengthZ] = size(InXYZFrames);

% Pre-allocate
rtrnCsTauXYZ = zeros(lengthX*lengthY, 1);

% Calc 
fprintf('\nProgress, calc Cxy(tau) by FFT: 000.0 [%%] | 00000.0 [sec]');
for iZ = 1:ZWindowSizeFrames % average over the window size
    startTime = tic;    
    
    % Calc autocovariance Cs(X,Y) = <I°(X,Y)I(X+X0,Y+Y0)> - <I>^2    
    subFrame = InXYZFrames(:, :, iZ); % extract subframe given by the window size
    subFrame = subFrame(:);
    
    CtTau = xcov(subFrame); % calc Ct(tau) as FT of the square of the power spectrum of the input signal (plus remove the contribution of the DC term)
    CtTau = CtTau(lengthX*lengthY:end); % remove first half of the function (in the negative domain)
    CtTauNormalized = CtTau./CtTau(1); % normalize to remove the dependence on the illumination intensity
    rtrnCsTauXYZ = rtrnCsTauXYZ + CtTauNormalized; % the normalized autocovariance for the given pixel XY and window along Z
    
    elapsedTime = toc(startTime);
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'); % delete previous progress line
    fprintf('%05.1f [%%] | %07.1f [sec]', (iZ/ZWindowSizeFrames)*100, (ZWindowSizeFrames-iZ)*elapsedTime);
end

rtrnCsTauXYZ = rtrnCsTauXYZ./ZWindowSizeFrames; % overage over the number of Cs(X,Y)

end

function rtrnCstTauXYZ = stFDLSIDirect(InXYZFrames, ZWindowSizeFrames, NumericalMethod, PixX1, PixY1, PixX2, PixY2)
% Calculate cross-correlation (for two pixel pairs in XY) of intensity signal

% Calc Ct(tau), i.e. the cross-covariance
switch(NumericalMethod)
    case 'fft' % Calc using FFT
        rtrnCstTauXYZ = stCrossCovarianceFFT(InXYZFrames, ZWindowSizeFrames, PixX1, PixY1, PixX2, PixY2);        
    case 'xcov' % calc using buit in matalb covariance xcov() function
        rtrnCstTauXYZ = stCrossCovarianceXcov(InXYZFrames, ZWindowSizeFrames, PixX1, PixY1, PixX2, PixY2);
    otherwise
        fprintf('\n\nUnsupported numerical method: %s\', NumericalMethod);
        error('Exit due to the above error!');
end

end

function rtrnCstTauXYZ = stCrossCovarianceFFT(InXYZFrames, ZWindowSizeFrames, PixX1, PixY1, PixX2, PixY2)
% Calculate cross-correlation (for two pixel pairs in XY) of intensity signal by Fourier Transform

[lengthX, lengthY, lengthZ] = size(InXYZFrames);
fitLengthZ = floor(lengthZ/ZWindowSizeFrames); % how many times the z window fits in all frames

% Pre-allocate
rtrnCstTauXYZ = zeros(fitLengthZ, ZWindowSizeFrames); % it can contain more than one Ct(tau9 curve as many as can be fitted in the size

% Calc 
fprintf('\nProgress, calc Ct(X1Y1, X2Y2, tau) by FFT: 000.0 [%%] | 00000.0 [sec]');
iZBegin = 1;
for iZ = 1:fitLengthZ % loop as many times as ZWindowSizeFrames fits in all available frames
    startTime = tic;    
    % Calc crosscovariance Ct(tau) = <I1°(t)I2(t+tau)> - <I1><I2>
    subFrame1 = InXYZFrames(PixX1, PixY1, iZBegin:(iZBegin+ZWindowSizeFrames-1)); % extract subframe given by the window size
    subFrame2 = InXYZFrames(PixX2, PixY2, iZBegin:(iZBegin+ZWindowSizeFrames-1)); % extract subframe given by the window size
    subFrame1 = subFrame1(:);
    subFrame2 = subFrame2(:);
    meanIntensity1 = sum(subFrame1, 'all')/ZWindowSizeFrames; % mean intensity    
    meanIntensity2 = sum(subFrame2, 'all')/ZWindowSizeFrames; % mean intensity
    subFrame1ZP = [subFrame1 - meanIntensity1; zeros(ZWindowSizeFrames,1)]; % add zero padding to cacl properly the cross-correlation
    subFrame2ZP = [subFrame2 - meanIntensity2; zeros(ZWindowSizeFrames,1)]; % add zero padding to cacl properly the cross-correlation
        
    % Calc cross-correlation by convolution --> f*g = F^-1{conj(F{f}).F{g}}
    Iw1 = conj(fft(subFrame1ZP)); % calc the conjugate of the Fourier Transform of the input intensity along temporal direction
    Iw2 = fft(subFrame2ZP); % calc the Fourier Transform of the input intensity along temporal direction    
    CtTau = ifft(Iw1.*Iw2); % calc Ct(tau) as FT of the input signals    
    CtTau = CtTau(1:ZWindowSizeFrames); % skips out the zero padding part
    
    rtrnCstTauXYZ(iZ, 1:ZWindowSizeFrames) = CtTau./max(CtTau, [], 'all'); % the crosscovariance for the given pixels X1Y1 and X2Y2 and window along Z
    
    iZBegin = iZ*ZWindowSizeFrames;
    elapsedTime = toc(startTime);
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'); % delete previous progress line
    fprintf('%05.1f [%%] | %07.1f [sec]', (iZ/fitLengthZ)*100, (fitLengthZ-iZ)*elapsedTime);
end

end

function rtrnCstTauXYZ = stCrossCovarianceXcov(InXYZFrames, ZWindowSizeFrames, PixX1, PixY1, PixX2, PixY2)
% Calculate cross-correlation (for two pixel pairs in XY) of intensity signal by matlab xcov() function

[lengthX, lengthY, lengthZ] = size(InXYZFrames);
fitLengthZ = floor(lengthZ/ZWindowSizeFrames); % how many times the z window fits in all frames

% Pre-allocate
rtrnCstTauXYZ = zeros(fitLengthZ, ZWindowSizeFrames); % it can contain more than one Ct(tau9 curve as many as can be fitted in the size

% Calc 
fprintf('\nProgress, calc Ct(X1Y1, X2Y2, tau) by xcov(): 000.0 [%%] | 00000.0 [sec]');
iZBegin = 1;
for iZ = 1:fitLengthZ % loop as many times as ZWindowSizeFrames fits in all available frames
    startTime = tic;    
    % Calc crosscovariance Ct(tau) = <I1°(t)I2(t+tau)> - <I1><I2>
    subFrame1 = InXYZFrames(PixX1, PixY1, iZBegin:(iZBegin+ZWindowSizeFrames-1)); % extract subframe given by the window size
    subFrame2 = InXYZFrames(PixX2, PixY2, iZBegin:(iZBegin+ZWindowSizeFrames-1)); % extract subframe given by the window size
    %subFrame = permute(subFrame, [3, 1, 2]);
    %meanIntensity1 = sum(subFrame1, 'all')/ZWindowSizeFrames; % mean intensity    
    %meanIntensity2 = sum(subFrame2, 'all')/ZWindowSizeFrames; % mean intensity
        
    % Calc cross-correlation by matlab's xcov() function
    CtTau = xcov(subFrame1(:), subFrame2(:)); % calc cross-covariance of Ct(r1, r2, tau)
    CtTau = CtTau(ZWindowSizeFrames:end); % take only the left part of the result (0, max)
    CtTauNormalized = CtTau./max(CtTau, [], 'all');
    rtrnCstTauXYZ(iZ, 1:ZWindowSizeFrames) = CtTauNormalized; % the crosscovariance for the given pixels X1Y1 and X2Y2 and window along Z
    
    iZBegin = iZ*ZWindowSizeFrames;
    elapsedTime = toc(startTime);
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'); % delete previous progress line
    fprintf('%05.1f [%%] | %07.1f [sec]', (iZ/fitLengthZ)*100, (fitLengthZ-iZ)*elapsedTime);
end

end

function rtrnCt = calcTheoryCtVTau(Tau, Velocity, WavelengthUm, NA)
% Calculate Ct(tau, v) with no diffusion (i.e. velocity >> diffusion), Ct(tau, v) = exp(-(M*v*tau)^2/len0^2), l0 = decorelation length (see eq. 9 in the paepr below)
% Hao Li et al, "Directly measuring absolute flow speed by frequency-domain laser speckle imaging", DOI link: https://doi.org/10.1364/OE.22.021079
% Velocity = direted flow velocity [Wavelength]/s (unit depends on the wavelength unit)
% Wavelength = illumintation wavelength (its unit will define the velocity unit)
% NA = numerical aperture
% Note: Magnification M cancels out

% Calc decorelation length
len0 = 0.41*WavelengthUm/NA;

% Calc Ct(tau, v)
rtrnCt = exp(-(Velocity.*Tau).^2./len0^2); % Gaussian velocity distribution
%rtrnCt = exp(-(Velocity.*Tau)./len0); % Lorentzian velocity distribution

end

function rtrnCt = calcTheoryCtV0VdTau(Tau, V0, Vd, WavelengthUm, NA)
% Calculate Ct(tau, v) with diffusion, i.e. Ct(tau, v0, vd), len0 = decorelation length (see eq. 13 in the paper below)
% Hao Li et al, "Directly measuring absolute flow speed by frequency-domain laser speckle imaging", DOI link: https://doi.org/10.1364/OE.22.021079
% V0 = mean velocity (directed velocity due to flow) [Wavelength]/s (unit depends on the wavelength unit)
% Vd = root mean square velocity (due to diffusion) [Wavelength]/s (unit depends on the wavelength unit)
% Wavelength = illumintation wavelength (its unit will define the velocity unit)
% NA = numerical aperture
% Note: Magnification M cancels out

% Calc decorelation length
len0 = 0.41*WavelengthUm/NA;

% Precalc
expr1 = len0^2 + (Vd.*Tau).^2;

% Calc Ct(tau, v0, vd)
Ct1 = exp(-(V0.*Tau).^2./expr1);
Ct2 = len0^3./expr1.^1.5 + (2*V0.^2.*Vd.^2.*len0.*Tau.^4)./expr1.^2.5;

rtrnCt = Ct1.*Ct2;

end

function saveLspData(DataLSP, InputFile, PixelXY, PixelWindowXY, CamExposureTime, FrameRate, WavelengthUm, NA)
% Save the processed LSP data

% Single XY pixel location to calc/show/save curves
pixX = PixelXY(2);
pixY = PixelXY(1);

XPixelWindow = PixelWindowXY(2);
YPixelWindow = PixelWindowXY(1);

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

% Save LSP Autocovariance --> Ct(Tau)
if ~isempty(DataLSP.CtTauXYZ)
    % Build base file name
    BaseFileName = [inputFileName sprintf('_fftLSA_Ct=(%d,%d)(%dx%d)', pixY, pixX, YPixelWindow, XPixelWindow)];
    
    % Get average (LSP Autocovariance) of Ct(Tau) curves in the respective window    
    numPoints = length(DataLSP.CtTauXYZ.Ctex);
    
    % Save Ct(tau) curve as csv .dat file
    txtFileName = [BaseFileName '.dat'];
    fileId = fopen(txtFileName, 'w'); % open the file for writing
    
    % Check if openning file was successful
    if (fileId == -1)
        error(['Writing to file failed! --> Filepath = ' txtFileName]);  % inform user about the error
    end
    
    % Write header    
    fprintf(fileId, 'tau,Ctex,CfitV0,CfitV0Vd\n');
    fprintf(fileId, '[s],[-],[-],[-]');
        
    % Write data    
    for i = 1:numPoints
        fprintf(fileId, '\n');        
        fprintf(fileId, '%f,%f,%f,%f', DataLSP.CtTauXYZ.Tau(i), DataLSP.CtTauXYZ.Ctex(i), DataLSP.CtTauXYZ.CtfitV0(i), DataLSP.CtTauXYZ.CtfitV0Vd(i));          
    end
    
    fclose(fileId);
end

% Save LSP PSD (Power Spectrum Density) --> save every first measurement in the center of the image for every global time point
if ~isempty(DataLSP.PSD)
    % Build base file name
    BaseFileName = [inputFileName sprintf('_fftLSA_PSD=(%d,%d)(%dx%d)', pixY, pixX, YPixelWindow, XPixelWindow)];
    
    % Get average (LSP Power SPectrum Density) of signal in the respective window
    numPoints = length(DataLSP.PSD.Amplitude);
    
    % Save PSD as csv .dat file
    txtFileName = [BaseFileName '.dat'];
    fileId = fopen(txtFileName, 'w'); % open the file for writing
    
    % Check if openning file was successful
    if (fileId == -1)
        error(['Writing to file failed! --> Filepath = ' txtFileName]);  % inform user about the error
    end
    
    % Write header    
    fprintf(fileId, 'freq[Hz],PSD[-]');
        
    % Write data    
    for i = 1:numPoints
        fprintf(fileId, '\n');        
        fprintf(fileId, '%f,%f', DataLSP.PSD.Freq(i), DataLSP.PSD.Amplitude(i));          
    end
    
    fclose(fileId);
end

% Save LSP Autocovariance fittted params --> V0, and V0+Vd
if ~isempty(DataLSP.CtTauXYZ.FitParms.SingleFlowModel) && ~isempty(DataLSP.CtTauXYZ.FitParms.FlowDiffModel)
    % Build base file name
    BaseFileName = [inputFileName sprintf('_fftLSA_V=(%d,%d)(%dx%d)', pixY, pixX, YPixelWindow, XPixelWindow)];
        
    % Save Ct(tau) curve as csv .dat file
    txtFileName = [BaseFileName '.dat'];
    fileId = fopen(txtFileName, 'w'); % open the file for writing
    
    % Check if openning file was successful
    if (fileId == -1)
        error(['Writing to file failed! --> Filepath = ' txtFileName]);  % inform user about the error
    end
    
    % Write header    
    fprintf(fileId, 'fit,value,lowerb,upperb\n');
    fprintf(fileId, 'Cfit,[mm/s],[mm/s],[mm/s]\n');
    
    % Write data     
    fprintf(fileId, 'V0(single):%f,%f,%f\n', DataLSP.CtTauXYZ.FitParms.SingleFlowModel.V0.value, DataLSP.CtTauXYZ.FitParms.SingleFlowModel.V0.lower, DataLSP.CtTauXYZ.FitParms.SingleFlowModel.V0.upper);
    fprintf(fileId, 'V0(fl+dif):%f,%f,%f\n', DataLSP.CtTauXYZ.FitParms.FlowDiffModel.V0.value, DataLSP.CtTauXYZ.FitParms.FlowDiffModel.V0.lower, DataLSP.CtTauXYZ.FitParms.FlowDiffModel.V0.upper);
    fprintf(fileId, 'Vd(fl+dif):%f,%f,%f', DataLSP.CtTauXYZ.FitParms.FlowDiffModel.Vd.value, DataLSP.CtTauXYZ.FitParms.FlowDiffModel.Vd.lower, DataLSP.CtTauXYZ.FitParms.FlowDiffModel.Vd.upper);
    
    fclose(fileId);
end

end
