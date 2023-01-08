%{
        Meta Learning for Extrapolating Nonlinear Dynamic Stochastic
        Systems

        Jashua Luna
        January 2023

        Script for evaluating trained model. 
%}

training_set    = get_data;                     % training set
validation_set  = get_data("ValidationData");   % validation set

[tx,ty]   = datagen(training_set,100,ws,nss,ovl,exl);    % sample from training set
[vx,vy] = datagen(validation_set,100,ws,nss,ovl,exl);    % sample from validation set

[tx,ty] = getbatch(tx,ty);
[vx,vy] = getbatch(vx,vy);

txh = inject_noise(tx(end-ws+1:end,:,:));
vxh = inject_noise(vx(end-ws+1:end,:,:));

ty_ = gen_predict(gen,tx,txh,ws,nss,ovl);
vy_ = gen_predict(gen,vx,vxh,ws,nss,ovl);

ty.y = gatext(squeeze(ty.y));
vy.y = gatext(squeeze(vy.y));

ty_ = gatext(ty_);
vy_ = gatext(vy_);

terror = mean(abs(ty.y-ty_),'all');
verror = mean(abs(vy.y-vy_),'all');

fprintf("training set average error: %.3f\n",terror);
fprintf("validation set average error: %.3f\n",verror);

figure
for i = 1:10
    subplot(10,2,2*i-1)
    plot(ty_(:,i));
    hold on 
    plot(ty.y(:,i));
    hold off 
    xline(ws-exl);
    
    subplot(10,2,2*i)
    plot(cumsum(ty_(:,i)));
    hold on 
    plot(cumsum(ty.y(:,i)));
    hold off
    xline(ws-exl);
end

figure
for i = 1:10
    subplot(10,2,2*i-1)
    plot(vy_(:,i));
    hold on 
    plot(vy.y(:,i));
    hold off 
    xline(ws-exl);

    subplot(10,2,2*i)
    plot(cumsum(vy_(:,i)));
    hold on 
    plot(cumsum(vy.y(:,i)));
    hold off
    xline(ws-exl);
end








%% Helper Functions

function [xbatch,ybatch] = getbatch(x,y)

    batchsize = length(x);
    
    xbatch      = zeros([size(x{1}) batchsize],'single');          % initialize xbatch
    ybatch.x    = zeros([size(y{1}.x) batchsize],'single');        % initialize ydata.x 
    ybatch.y    = zeros([size(y{1}.y) batchsize],'single');        % initialize ydata.y
    ybatch.y_dt = zeros([size(y{1}.y_dt) batchsize],'single');     % initialize ydata.y_dt
    
    for i = 1:batchsize    
        xbatch(:,i)         = x{i};             % load xdata into array
        ybatch.x(:,:,i)     = y{i}.x;           % load subsampled, reweighted, and detrended data into array
        ybatch.y(:,i)       = y{i}.y;           % load reference output into array
        ybatch.y_dt(:,i)    = y{i}.y_dt;        % load scaled, reweighted, detrended reference output into array
    end
    
    ybatch.x    = gpudl(ybatch.x,'');       % make ybatch.x a traced dl array and load onto gpu memory
    ybatch.y    = gpudl(ybatch.y,'');       % make ybatch.y a traced dl array and load onto gpu memory
    ybatch.y_dt = gpudl(ybatch.y_dt,'');    % make ybatch.y_dt a traced dl array and load onto gpu memory

end

function [xdata,ydata] = datagen(data,ns,ws,nss,ovl,exl)

    [xdata,ydata] = deal(cell(ns,1)); % empty cell arrays
    
    xWin    = ws + (nss-1)*(ws - ovl); % input size
    xlocs   = [-xWin+1:0];             % template xlocs
    ylocs   = [-ws+1:0]+exl;                % template ylocs                    
    
    locs = randi([xWin size(data,1)-exl-1],[ns 1]);  % sampling locations
    
    for i = 1:ns                                                                % loop through samples
        xdata{i}        = data(locs(i) + xlocs,5) - data(locs(i) + xlocs,2);    % sample xdata as closing price - open price 
        
        ydata{i}.x      = scale_detrend_reweight(xdata{i},true);                % subsample,scale, detrend, reweight xdata
        ydata{i}.y      = data(locs(i) + ylocs,5) - data(locs(i) + ylocs,2);    % sample ydata as closing price - open price
        ydata{i}.y      = scalinglayer(ydata{i}.y);                             % scale ydata
        
        ydata{i}.y_dt   = reweight_detrend(ydata{i}.y,false);                   % detrend, reweight ydata
        ydata{i}.y      = reweight(ydata{i}.y);                                 % reweight ydata
    end
    
    function y = scale_detrend_reweight(x,subsampleflag)  
        % function for scaling, reweighting and detrending data
        
        y = scalinglayer(x);                                 % scale data
        y = reweight_detrend(y,subsampleflag,ws,ovl,nss);    % reweight and detrend data
    end
end

function [y,ScaleFactor] = scalinglayer(x)
    % layer for scaling and subsampling input data
    
    ScaleFactor = zeros(size(x,3),1);
    for i = 1:length(ScaleFactor)
        if ndims(x)==3
            ScaleFactor(i) = confidence_bounds(x(:,:,i),0.95);    % determine scale factor based on 95% confidence interval
        else
            ScaleFactor(i) = confidence_bounds(x(:,i),0.95);    % determine scale factor based on 95% confidence interval
        end
    end
    
    if ndims(x)==3
        ScaleFactor = permute(ScaleFactor,[2 3 1]);
    else
        ScaleFactor = permute(ScaleFactor,[2 1]);
    end
    
    y = x./ScaleFactor;                                   

    y(isnan(y)) = 0;
    y(isinf(y)) = 0;
            
    function sf = confidence_bounds(sample,ci)
        % function for determining scale factor based on confidence interval
        
        [mu,sig] = normfit(sample);              % fit standard distribution to data
        x1 = linspace(min(sample),max(sample),1e3);    % make points at which normcdf will be evaluated
        p = normcdf(x1,mu,sig);              % evaluate norm cdf
        sf = x1(p>=ci);                      % isolate confidence region
        try
            sf = sf(1);                         % take just the first point
        catch
            sf = x1(end);
        end
    end
end

function y = reweight(x)
    % function for reweighting samples
    
    y = huber_reweight(x);
    
    function xbatch = huber_reweight(xbatch)
        % function for reweighting xbatch with huber weight function
        % xbatch is in format TC
        
        sf  = mad(xbatch,1,1)/0.6745;               % compute scale factor as median absolute deviation divided by 0.6745
        r   = abs(xbatch - median(xbatch,1))./sf;    % compute residual as abs delta x - median(x) divided by sf
    
        w = huber_weight(r,1.547);       % get huber weights
        xbatch = w.*xbatch;             % reweight samples
    end
    
    function weights = huber_weight(x,c)
        % function for determining huber weights
        
        assert(ndims(x)==2,"X must have two dimensions");
    
        weights = ones(size(x));                   % weights where x < c = 1
        weights(x > c) = c./x(x>c);                 % c/x otherwise     
    end
end

function y = reweight_detrend(x,subsampleflag,ws,ovl,nss)

    % function for generating reweighted-detrended samples 
    % where x is a two dimensional array TB 
    
    if subsampleflag            % if sample should be subsampled
        x = subsample(x);       % subsample
    end
    
    x = reweight(x);                                        % subsample and reweight x 
    y = waveletlayer(x);                                    % obtain the waveletlayer output of x
    y = inverse_waveletlayer(y,zeros(size(x),'single'));    % inverse wavelet layer y
    
    % Hotfix -> if sample created NaN or Inf, ignore detrend operation
%     if all(isnan(y),'all');y=x;end
    
    function y = inverse_waveletlayer(x,y)
        % function for recovering signal from cwt of signal
        
        % HOTFIX -> Some samples were creating NaN and Inf values 
%         try
%             validateattributes(x,"numeric","finite");
%         catch
%             y = NaN;
%             return
%         end
        
        for i = 1:size(x,3)                 % loop through subsamples
            y(:,i) = icwt(x(:,:,i));        % inverse continous wavelet transform
        end
    end
    
    function x1 = subsample(x)
        % function specific function for subsampling x : handles one sample at a time
    
        x1 = zeros([ws nss],'single');    % initialize subsampled x as zeros
        sslocs = [1:ws];                  % init subsampling locs
    
        for i = 1:nss
            x1(:,i) = x(sslocs);                                        % subsample x
            sslocs = sslocs + (ws - ovl); % increment locs
        end              
    end
end

function y = waveletlayer(x)
    persistent filter_idx psi_fvec
    if isempty(filter_idx)
        wavelet = cwtfilterbank('SignalLength',size(x,1),'Boundary','periodic');  % create a filter wavlet bank
        [psi_fvec,filter_idx] = cwtfilters2array(wavelet); 
        psi_fvec = gpuArray(psi_fvec);
        filter_idx = gpuArray(filter_idx);
    end
    
    y = dlcwt(x,psi_fvec,filter_idx,'DataFormat','TCB');      % Wavelet transform subsampled sets
    y = permute(y,[1 4 2 3]);                                   % dlcwt output is SCBT where S is filter dilation, permute to STCB
end