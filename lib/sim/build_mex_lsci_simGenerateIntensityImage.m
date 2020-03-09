% Generate mex function
clear

% Declare variables type

% Particles var
Particles = struct([]);
Particles(1).x0 = double(0); % store old X position
Particles(1).y0 = double(0); % store old Y position
Particles(1).z0 = double(0); % store old Z position
    
Particles(1).x = double(0); % new X position
Particles(1).y = double(0); % new Y position
Particles(1).z = double(0); % new Z position
Particles(2).Dx = double(0);
Particles(2).Dy = double(0);
Particles(2).Dz = double(0);
Particles(1).Ufield = complex(1.0); % the init amplitude and phase of the electric field for this particle

Particles(2).x0 = double(0); % store old X position
Particles(2).y0 = double(0); % store old Y position
Particles(2).z0 = double(0); % store old Z position
    
Particles(2).x = double(0); % new X position
Particles(2).y = double(0); % new Y position
Particles(2).z = double(0); % new Z position
Particles(2).Dx = double(0);
Particles(2).Dy = double(0);
Particles(2).Dz = double(0);
Particles(2).Ufield = complex(1.0); % the init amplitude and phase of the electric field for this particle

typeParticles = coder.typeof(Particles, [1 Inf], [false true]);

% ImageHeightPx
ImageHeightPx = uint32(1);
typeImageHeightPx = coder.typeof(ImageHeightPx, [1 1], [false false]);

% ImageWidthPx
ImageWidthPx = uint32(1);
typeImageWidthPx = coder.typeof(ImageWidthPx, [1 1], [false false]);

% LaserWavelengthUm
LaserWavelengthUm = double(0);
typeLaserWavelengthUm = coder.typeof(LaserWavelengthUm, [1 1], [false false]);

% RSCoreLookUpTbl
RSCoreLookUpTbl = complex(1.0);
typeRSCoreLookUpTbl = coder.typeof(RSCoreLookUpTbl, [Inf Inf], [true true]);

% ObservationPlaneXYZUm
ObservationPlaneXYZUm = struct();
ObservationPlaneXYZUm.X = coder.typeof(zeros(2,2), [Inf Inf], [true true]);
ObservationPlaneXYZUm.Y = coder.typeof(zeros(2,2), [Inf Inf], [true true]);
ObservationPlaneXYZUm.Z = double(0);
typeObservationPlaneXYZUm = coder.typeof(ObservationPlaneXYZUm);

% StepSizeXYZUm
StepSizeXYZUm = double(0);
typeStepSizeXYZUm = coder.typeof(StepSizeXYZUm);
    
codegen -config:mex lsci_simGenerateIntensityImage -args {typeParticles, typeImageHeightPx, typeImageWidthPx, typeLaserWavelengthUm, typeRSCoreLookUpTbl, typeObservationPlaneXYZUm, typeStepSizeXYZUm} -nargout 1