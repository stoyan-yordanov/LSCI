function scr_plotIntensityVsFrames(HeaderLines, FrameRate, XRange, XPlotType, YPlotType, PlotLegendLabels, PlotColors, PlotMarkers)
% Plotting Intensity in a pixel vs Frame number (or time)

% Read settings value from the cli (command line) input
% Get files from the current directory according to the file filter
strFileFilter = '*.*';
fileList = sys_GetDirectoryFileList(strFileFilter);  %get files only with the given extension

% Choose file(s) from file list
fileList = sys_ChooseFilesFromFileList(fileList);

% First create the figure
figHandle = -1;
if isempty(figHandle) || figHandle < 1
    figPos = [1000 200 260 200];
    hFig = figure('Color', 'w', 'Position', figPos); % create new figure
    axes();
else
    hFig = figure(figHandle); % activate fig with this handle
end

% Data structure
data = struct([]);
data(1).X = [];
data(1).Y = [];
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
        
        xyBuffer = sscanf(strLine, '%f,%f', [2, 1]); % get the current values in a buffer
        data(k).X(i - HeaderLines) = xyBuffer(1);
        data(k).Y(i - HeaderLines) = xyBuffer(2);
    end
    
    % Find the max of the peak
    [data(k).peakIntensity, indx] = max(data(k).Y);
    data(k).peakFrame = data(k).X(indx);    
    
    % Plot data
    %plLabel = [PlotLabels{k} '(\lambda_{max} = ' sprintf('%.1f', data(k).peakWavelength) ' nm)'];
    switch(XPlotType)
        case 'Frame'
            xLabelTxt = 'Frame #';
    	case 'Time'
            xLabelTxt = 'Time [s]';
            data(k).X = (1/FrameRate) * data(k).X; % frame num to time conversion
        otherwise
            fprintf('\nUnsupported X plot type --> XPlotType = %s', XPlotType);
            error('Exit due to the above error!');
    end
    
    switch(YPlotType)
        case 'Intensity'
            yLabelTxt = 'Intensity [a.u.]';
    	case 'Intensity2' % not used at the moment
            yLabelTxt = 'Intensity [a.u.]';
            data(k).Y = data(k).Y; % data conversion
        otherwise
            fprintf('\nUnsupported Y plot type --> YPlotType = %s', YPlotType);
            error('Exit due to the above error!');
    end
    
    plot(gca, data(k).X, data(k).Y, PlotMarkers{k}, 'DisplayName', PlotLegendLabels{k}, 'LineWidth', 1.0, 'Color', PlotColors{k});
    % plot(gca, data(k).X, data(k).Y, PlotMarkers{k}, 'DisplayName', PlotLabels{k}, 'LineWidth', 1.5, 'Color', PlotColors{k}, 'MarkerSize', 6, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0 0.5 1]);
end

% Set plot to a box
set(gca, 'FontSize', 8); % set font size
set(gca, 'LabelFontSizeMultiplier', 1); % multiplier for the font size of axis labels
set(gca, 'TitleFontSizeMultiplier', 1); % multiplier for the font size of titles
set(gca, 'TitleFontWeight', 'normal'); % normal/bold
set(gca, 'Box', 'on');

% Set XY limits
if isempty(XRange)
    %xrange = xlim();
    %xlim(gca, [min(data(1).Y), xrange(2)]);
else
    xlim(gca, XRange);
end

yrange = ylim();
ylim(gca, [yrange(1), yrange(2)]);

% Set axes
title(gca, '');
xlabel(gca, xLabelTxt);
ylabel(gca, yLabelTxt);
%title(sprintf('Velocity vs Frame/Time'));

% Turn on legend
if isempty(PlotLegendLabels) || length(PlotLegendLabels) <= 1
    legend('off');
else
    legend('Location', 'northwest');
end

hold off;

% Tighten plot - reduces white space as much as possible
plots_TightenAxes(gca);

% Save graph
plots_SavePlotAsImage2('Graph_', hFig, 'png', '-opengl', [YPlotType 'Vs' XPlotType]);

end

