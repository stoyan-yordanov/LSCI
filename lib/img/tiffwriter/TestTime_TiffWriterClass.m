clear all;close all; clc; fclose all;
%generate some data
nFrames = 1000;
IM = imread('landOcean.jpg');
IM = uint16(sum(IM,3));
IM = IM(101:640, 961:1680);
IM = IM-min(IM(:));
IM=IM*(2^15/max(IM(:)));
IM = repmat(IM, [1,1,nFrames])+randi((2^15)-1, [size(IM,1), size(IM,2), nFrames], 'uint16');
S = (numel(IM)/nFrames*2)/2^20;

%https://nl.mathworks.com/matlabcentral/fileexchange/35684-multipage-tiff-stack
%imread writespeed
methods = {'imwrite','tifflib','fTIF'};
for M = 1:length(methods)
    method = methods{M};
    
    %timing vector
    t = nan(1,size(IM,3)+1);
    
    %file
    filename = [method,'.tif'];
    if exist(filename,'file'), delete(filename);end
    switch method
        case 'imwrite'
            tic;
            t(1)=0;
            imwrite(IM(:,:,1),filename,'Compression','none');
            t(2)=toc;
            for ct = 2:size(IM,3)
                imwrite(IM(:,:,ct),filename,'WriteMode','append','Compression','none');
                t(ct+1)=toc;
            end
        case 'tifflib'
            t(1)=0;
            tic;
            tf = Tiff(filename,'w');
            for ct = 1:size(IM,3)
                if ct>1,tf.writeDirectory;end
                tf.setTag('Photometric',Tiff.Photometric.MinIsBlack);
                tf.setTag('Compression',Tiff.Compression.None);
                tf.setTag('BitsPerSample',16);
                tf.setTag('SamplesPerPixel',1);
                tf.setTag('SampleFormat',Tiff.SampleFormat.UInt);
                tf.setTag('ExtraSamples',Tiff.ExtraSamples.Unspecified);
                tf.setTag('ImageLength',size(IM,1));
                tf.setTag('ImageWidth',size(IM,2));
                tf.setTag('PlanarConfiguration',Tiff.PlanarConfiguration.Chunky);
                tf.setTag('ImageDescription',sprintf('ImageJ=1.51j\nchannels=%.0f',size(IM,3)));
                tf.write(IM(:,:,ct));
                t(ct)=toc;
            end
            tf.close();
        case 'fTIF'
            t(1)=0;
            tic
            fTIF = TiffWriterClass(filename, 37.8, 0);
            for ct = 1:size(IM,3)
                fTIF.WriteImage(IM(:,:,ct));
                t(ct)=toc;
            end
            tic
            fTIF.close;
            fprintf(1,'closing the tif file by fTIF.close took %.5f seconds\n',toc);
        otherwise
            error('unknown method')
    end
    
    S = (size(IM,1)*size(IM,2)*2)/2^20; %MB/frame
    y = S./diff(t);
    subplot(1,length(methods),M)
    plot([1:size(IM,3)],y);
    title(sprintf('Writing with %s; mean = %.2f MB/s',method,mean(y,'omitnan')))
    ylabel('Writing speed (MB/s)')
    xlabel('Frame');
    drawnow;
end

f=gcf;
f.Position=[1000 918 1175 420];
saveas(f,'example.png')
