function rtrnIntensityImage = lsci_simGenerateIntensityImage(Particles, ImageHeightPx, ImageWidthPx, LaserWavelengthUm, RSCoreLookUpTbl, ObservationPlaneXYZUm, StepSizeXYZUm)
% Generate Intensity field diffraction pattern in XY plane from particle positions by E-field wave propagation

numberParticles = length(Particles);
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
numSubSteps = floor(2*maxdr/LaserWavelengthUm + 1); % number of steps to walk between old and new position, each step moves a length equal half the wavelength
dx = dx./numSubSteps; % fine dx step
dy = dy./numSubSteps; % fine dy step
dz = dz./numSubSteps; % fine dz step

% Calc intensity image
for iSubStep = 1:numSubSteps
    tmpUfiledImage = complex(zeros(ImageHeightPx, ImageWidthPx)); % reset
    
    % Calc E-field image
    parfor i = 1:numberParticles % loop thorough the particles

        % Fine propagation to calc Electrical field diffraction pattern        
        %tmpParticlePosXUm = Particles(i).x0 + dx(i)*iSubStep; % update position
        %tmpParticlePosYUm = Particles(i).y0 + dy(i)*iSubStep; % update position
        %tmpParticlePosZUm = Particles(i).z0 + dz(i)*iSubStep; % update position    
        
        % Calc XYZ distance particle-pixel in Matrix form
        Rx = Particles(i).x0 - ObservationPlaneXYZUm.X; % matrix/vector form
        Ry = Particles(i).y0 - ObservationPlaneXYZUm.Y; % matrix/vector form
        Rz = Particles(i).z0 - ObservationPlaneXYZUm.Z; % not a matrix/vector     
        R = sqrt(Rx.^2 + Ry.^2 + Rz.^2); % max absolute distance source-point
        
        % Calc indexes for the lookup table value in Matrix form
        iR = ceil(R./StepSizeXYZUm);
        iRz = ceil(Rz./StepSizeXYZUm);
        
        % Calc Ufield as a function of distance R matrix
        if Particles(i).Ufield == 1
            % Assumes Ufield = 1 => no multiplication necessary
            Upix = reshape(RSCoreLookUpTbl(iR(:), iRz), [ImageHeightPx, ImageWidthPx]); % precalc the RS core array for the given particle
        else
            Upix = Particles(i).Ufield .* reshape(RSCoreLookUpTbl(iR(:), iRz), [ImageHeightPx, ImageWidthPx]); % precalc the RS core array for the given particle
        end
        
        % Assign intensity to the given pixel based on the particle position --> vectorized form
        tmpUfiledImage = tmpUfiledImage + Upix;         
    end
    
    % Add E-field image to the Intensity image
    rtrnIntensityImage = rtrnIntensityImage + abs(tmpUfiledImage).^2;
    
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
