function rtrnFileList = lsci_sysChooseFilesFromFileList(FileList)
% Returns the file(s) name(s) to be processed in the current directory

fileCount = size(FileList, 1);

% List the files in the current directory
fprintf('\nList of the available files in the current directory (# %d):\n', fileCount);

for i = 1:fileCount
    fprintf(' (%d) %s\n', i, FileList{i, 1});
end

% Choose the file(s) to be processed
promptFileNameNumbers = '\nChoose file(s) to process from the list of the files above\n --> Select file(s) by typing the number in (..)\n --> Separate multi-files by space/comma\n --> Empty will processes all files\n fileNameNumbers = ';
fileNameNumbers = input(promptFileNameNumbers, 's'); % input is expected to be string ('s' option) and not expression

% Extract file numbers from the string
fileNameNumbers = str2num(fileNameNumbers); %convert str to num matrix vector
% fileNameNumbers = sscanf(fileNameNumbers, '%d'); %convert str to num matrix vector

% Extract file numbers from the string
% cellFileNameNumbers = regexp(fileNameNumbers, '[0-9]', 'Match'); %convert str to cell array
% fileNameNumbers = zeros(1);
% for i= 1:length(cellFileNameNumbers)
%   if ~isempty(cellFileNameNumbers{i})
%       fileNameNumbers(i) = str2num(cellFileNameNumbers{i});
%   end
% end

% Re-index to process only the selected files
if (~isempty(fileNameNumbers))
    rtrnFileList = cell(length(fileNameNumbers), 1);    
    for i = 1:length(fileNameNumbers)
        indx = fileNameNumbers(i); % extract the next file number
        rtrnFileList{i, 1} = FileList{indx, 1}; % get the file name corresponding to the selected file number
    end
else
    rtrnFileList = FileList; % get the file name corresponding to the selected file number
end

% Get the number of files to be processed
fileCount = size(rtrnFileList, 1);

% Check if we have a file to process
if (isempty(rtrnFileList) || fileCount < 1)
    error('\nError in sys_ChooseFilesFromFileList --> No files to process!\n');
else
    fprintf('\nYou are about to process %d file(s):\n', fileCount);
    
    % Print the files to be processed
    for i = 1:fileCount
        fprintf(' (%d) %s\n', i, rtrnFileList{i, 1});
    end
end

end