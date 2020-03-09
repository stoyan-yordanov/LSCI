function rtrnIntensityImage = lsci_simGenerateDiffusiveIntensityImage(Particles, StaticScatters, ImageHeightPx, ImageWidthPx, LaserWavelengthUm, RSCoreLookUpTbl, ObservationPlaneXYZUm, StepSizeXYZUm)
% Generate Intensity field diffraction pattern in XY plane from particle positions by E-field wave propagation

numberParticles = length(Particles);
numberStaticScatters = length(StaticScatters);
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
    
    % Calc E-field
    parfor i = 1:numberParticles % loop through the particles
        
        % Init U-field of particle i
        Ufield = complex(Particles(i).Ufield);
                
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
%             Ufield = Ufield + 0.1 .* Particles(j).Ufield .* RSCoreLookUpTbl(ipR, ipRz); % 10% light scattering coefficient
%             %Ufield = Ufield + Particles(j).Ufield .* RSCoreLookUpTbl(ipR, ipRz); % 100% light scattering coefficient
%         end
        
        % Calc Z distance particle-pixel
        Rz = Particles(i).z0 - ObservationPlaneXYZUm.Z; % observation plane is along Z
        iRz = ceil(Rz./StepSizeXYZUm);
        
        % Calc XY distance particle-pixel in Matrix form
        Rx = Particles(i).x0 - ObservationPlaneXYZUm.X;
        Ry = Particles(i).y0 - ObservationPlaneXYZUm.Y;
        %Rz = Particles(i).z0 - observationPlaneXYZUm.Z;        
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
    rtrnIntensityImage = rtrnIntensityImage + abs(tmpUfieldImage).^2;
    
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
