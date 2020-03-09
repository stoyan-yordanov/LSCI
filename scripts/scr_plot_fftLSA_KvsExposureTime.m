function scr_plot_fftLSA_KvsExposureTime(HeaderLines, FileColumnsToPlot, XPlotLabel, YPlotLabel, PlotLegendLabels, PlotColors, PlotMarkers)
% Plotting LSP Contrast K (experimental and fitted) vs Exposure time

% Read settings value from the cli (command line) input
% Get files from the current directory according to the file filter
strFileFilter = '*.*';
fileList = sys_GetDirectoryFileList(strFileFilter);  %get files only with the given extension

% Choose file(s) from file list
fileList = sys_ChooseFilesFromFileList(fileList);

% First create the figure
figHandle = -1;
if isempty(figHandle) || figHandle < 1
    figPos = [1000 200 260 240];
    hFig = figure('Color', 'w', 'Position', figPos); % create new figure
    axes();
else
    hFig = figure(figHandle); % activate fig with this handle
end

% Data structure
data = struct([]);
data(1).X1 = [];
data(1).Y2 = [];
data(1).Y3 = [];
data(1).peakIntensity = 0;
data(1).peakFrame = 0;

% Plot data
hold on;

for k = 1:size(fileList, 1)
    
    fileName = fileList{k, 1}; % get current file name
    
    % Load data
    txtBuffer = sys_LoadTextFile(fileName);
    txtbLength = size(txtBuffer{1}, 1);
    
    % Extract the data from the file
    HeaderOffset = HeaderLines + 1;
    for i = HeaderOffset:txtbLength        
        strLine = txtBuffer{1}{i}; % load txt line
        
        % Exit if empty line occured
        if strncmp(strLine, '', 1) || strncmp(strLine, '\n', 1) || strncmp(strLine, '\r\n', 2)
            break;
        end
        
        xyBuffer = sscanf(strLine, '%f,%f,%f,%f', [1 4]); % get the current values in a buffer
        data(k).X1(i - HeaderLines) = xyBuffer(1);
        data(k).Y2(i - HeaderLines) = xyBuffer(2);
        data(k).Y3(i - HeaderLines) = xyBuffer(3);
    end
    
    % Plot data
    for j = 1:length(FileColumnsToPlot)
        %plLabel = [PlotLabels{k} '(\lambda_{max} = ' sprintf('%.1f', data(k).peakWavelength) ' nm)'];
        fieldName = ['Y' FileColumnsToPlot{j}]; % generate the field name
        plot(gca, data(k).X1, data(k).(fieldName), PlotMarkers{k, j}, 'DisplayName', PlotLegendLabels{k, j}, 'LineWidth', 1.0, 'Color', PlotColors{k, j});
    end
end

% Set plot to a box
set(gca, 'FontSize', 8); % set font size
set(gca, 'LabelFontSizeMultiplier', 1); % multiplier for the font size of axis labels
set(gca, 'TitleFontSizeMultiplier', 1); % multiplier for the font size of titles
set(gca, 'TitleFontWeight', 'normal'); % normal/bold
set(gca, 'Box', 'on');
set(gca, 'XScale', 'log'); % axis scaling (linear or log)

% Set XY limits
xrange = xlim();
xlim(gca, [xrange(1), xrange(2)]);

yrange = ylim();
ylim(gca, [yrange(1), yrange(2)]);
%ylim(gca, [0, 1]);

% Set axes
if ~isempty(XPlotLabel)
    xLabelTxt = XPlotLabel;
else
    xLabelTxt = 'Exposure Time [s]';
end

if ~isempty(YPlotLabel)
    yLabelTxt = YPlotLabel;
else
    yLabelTxt = 'Contrast K';
end

title(gca, '');
xlabel(gca, xLabelTxt);
ylabel(gca, yLabelTxt);
%title(sprintf(''));

% Turn on legend
if isempty(PlotLegendLabels) || length(PlotLegendLabels) <= 1
    legend('off');
else
    legend('Location', 'northeast');
end

hold off;

% Tighten plot - reduces white space as much as possible
plots_TightenAxes(gca);

% Save graph
plots_SavePlotAsImage2('Graph_', hFig, 'png', '-opengl', ['K' 'vs' 'ExposureTime']);

end

