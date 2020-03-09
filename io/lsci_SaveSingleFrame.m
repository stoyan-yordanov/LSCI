function lsci_SaveSingleFrame(OutputXYZFrame, BaseFileName, WriteMode)
% Save single image frame to file
% Function parameters:
% OutputXYZFrame = output data frames to be saved
% BaseFileName = file name of the output data (without the extension)
% WriteMode = 'append' (append to file) | 'overwrite' (overwrite file)

% Calc normalization factor
maxIt = max(OutputXYZFrame, [], 'all'); % global noramlization factor is the same for each frame

% Save processed data as tiff
fileNameAndPath = fullfile([BaseFileName '.tiff']); % Assemble tiff file name

OutputXYZFrame = OutputXYZFrame ./ maxIt; % normalize to 1
cInt = 2^16 - 1; % cInt = 255; % coefficient to convert to 8 or 16 bit integer
OutputXYZFrame = uint16(cInt.*OutputXYZFrame); % convert to 8 or 16 bit depth

imwrite(OutputXYZFrame, fileNameAndPath, 'tiff', 'Compression', 'packbits', 'WriteMode', WriteMode); % save Laser Speckle 3D stack

end
