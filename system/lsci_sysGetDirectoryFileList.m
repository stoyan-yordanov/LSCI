function rtrnFileList = lsci_sysGetDirectoryFileList(FileDirFilter)
% Get files from the current directory or directory specified from the filter

fileList = ls(FileDirFilter);  % get file list with extension indicated with the file filter
fileCount = size(fileList, 1);  % get the number of files to be processed

% Convert to cell array to remove the white space padding of Matlab
rtrnFileList = cell(fileCount, 1); % create cell array with the respective size
for i = 1:fileCount
    rtrnFileList{i, 1} = strtrim(fileList(i, :));
end

% Check if we have files to process
if (fileCount < 1)
    error('Error --> No files found in the current directory to be processed!');
end

end

