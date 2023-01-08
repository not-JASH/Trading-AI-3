%{
        Meta Learning for Extrapolating Nonlinear Dynamic Stochastic
        Systems

        Jashua Luna
        December 2022

        No objects.
%}

%% Global variables
getworkers;

% 0 5 10 20 40 80 160

efr = .2;

ws = 80;
exl = floor(efr*ws);
ovl = 30;
nss = 30;

[lrg,lrd] = deal(1e-3);

ns = 1e4;
bs = 50 ;
nvs = 15;

%% Initialize variables

gen = generator([],ws,exl,ovl,nss,bs,lrg);
gen = gen.weights;  % keep just the weights

disc = discriminator([],ws,lrd);
disc = disc.weights;    % keep just the weights

data = get_data;

[xdata,ydata] = deal(cell(ns,1));

parfor i = 1:ns
    [xdata(i),ydata(i)] = datagen(data,1,ws,nss,ovl,exl);
end

[rdl.avg_g,rdl.avg_sqg,pl.avg_g,pl.avg_sqg,idl.avg_g,idl.avg_sqg,e.avg_g,e.avg_sqg,disc_avg_g,disc_avg_sqg] = deal([]);

%% Training Loop

epoch = 1;
iter = 1;
total_time = 0;
iteration_time = 0;
avg_e = 1;

fprintf("Starting Training Loop\n");    

while avg_e > 0.08 
    tic 

    shuffle_locs = randperm(ns);
    batchlocs = [1:bs];

    while ~isempty(batchlocs)
        progressbar(batchlocs(end)/ns);

        [xbatch,ybatch] = getbatch(xdata(shuffle_locs(batchlocs)),ydata(shuffle_locs(batchlocs)));
        [grad_gen,grad_disc] = dlfeval(@modelgradients,gen,disc,xbatch,ybatch,ws,nss,ovl);

        % update reweight detrend layer parameters
        [gen.RDL,rdl.avg_g,rdl.avg_sqg] = adamupdate(gen.RDL,grad_gen.RDL,rdl.avg_g,rdl.avg_sqg,iter,lrg,1-1e-2,1-1e-4);

        % update prediction layer parameters
        [gen.PL,pl.avg_g,pl.avg_sqg] = adamupdate(gen.PL,grad_gen.PL,pl.avg_g,pl.avg_sqg,iter,lrg,1-1e-2,1-1e-4);

        % update inverse-detrend layer parameters
        [gen.IDL,idl.avg_g,idl.avg_sqg] = adamupdate(gen.IDL,grad_gen.IDL,idl.avg_g,idl.avg_sqg,iter,lrg,1-1e-2,1-1e-4);

        % update encoder layer parameters
        [gen.E,e.avg_g,e.avg_sqg] = adamupdate(gen.E,grad_gen.E,e.avg_g,e.avg_sqg,iter,lrg,1-1e-2,1-1e-4);

        % update discriminator layer parameters
        [disc,disc_avg_g,disc_avg_sqg] = adamupdate(disc,grad_disc,disc_avg_g,disc_avg_sqg,iter,lrd,1-1e-2,1-1e-4);

        batchlocs = batchlocs + bs;
        batchlocs(batchlocs>ns) = [];
        iter = iter+1;
    end

    % evaluate performance
    eval_locs = randi(ns,[nvs 1]);
    [validx,validy] = getbatch(xdata(eval_locs),ydata(eval_locs));
    validxh = inject_noise(inject_noise(validx(end-ws+1:end,:,:)));
    y = gatext(gen_predict(gen,validx,validxh,ws,nss,ovl));
    Y = squeeze(gatext(validy.y));
    avg_e = mean(abs(Y-y),'all');
    
    fprintf("Average Error: %.3f\n",avg_e); 
    
    for i = 1:nvs
        subplot(nvs,2,2*i-1)
        plot(Y(:,i));
        hold on 
        plot(y(:,i));
        hold off
        xline(ws-exl)

        subplot(nvs,2,2*i)
        plot(cumsum(Y(:,i)));
        hold on 
        plot(cumsum(y(:,i)));
        hold off
        xline(ws-exl);
    end
    
    f = getframe;  

    iteration_time = toc;                           % how long did this iteration take
    total_time = iteration_time + total_time;       % add iteration time to total time
    [d,h,m,s] = gettimestats(total_time);           % calculate time in days hours minutes seconds
    fprintf("Epoch %d Complete, Time Elapsed: %.2f s\n",epoch,iteration_time);   % display iteration info
    fprintf("Total Time Elapsed: %s:%s:%s:%s\n\n",d,h,m,s);                         % display training time info

    if rem(epoch,10) == 0
        save(sprintf("checkpointsave_%d.mat",epoch));
    end

    epoch = epoch+1;    
end

[err,verr] = evaluate_generator(xdata,ydata,gen,ws,nss,ovl,bs,exl);


%% Functions 

function [error,valid_error] = evaluate_generator(xdata,ydata,gen,ws,nss,ovl,bs,exl)

    nevalsamples = 5e2;

    error = zeros(length(xdata)/bs,1);
    batchlocs = [1:bs];
    i = 1;

    while ~isempty(batchlocs)
        [xbatch,ybatch] = getbatch(xdata(batchlocs),ydata(batchlocs));
        xh = inject_noise(xbatch(end-ws+1:end,:,:));
        y = gen_predict(gen,xbatch,xh,ws,nss,ovl);
        error(i) = mean(abs(gatext(squeeze(ybatch.y))-gatext(y)),'all');
        i = i+1;
        batchlocs = batchlocs + bs;
        batchlocs(batchlocs>length(xdata)) = [];
    end

    fprintf("Average error on training set: %.3f\n",mean(error));
    
    data = get_data;
    [validx,validy] = datagen(data,nevalsamples,ws,nss,ovl,exl);

    valid_error = zeros(nevalsamples/bs,1);
    batchlocs = [1:bs];
    i = 1;

    while ~isempty(batchlocs)
        [xbatch,ybatch] = getbatch(validx(batchlocs),validy(batchlocs));
        xh = inject_noise(xbatch(end-ws+1:end,:,:));
        y = gen_predict(gen,xbatch,xh,ws,nss,ovl);
        valid_error(i) = mean(abs(gatext(squeeze(ybatch.y))-gatext(y)),'all');
        i = i+1;
        batchlocs = batchlocs + bs;
        batchlocs(batchlocs>nevalsamples) = [];
    end

    fprintf("Average error on validation set: %.3f\n",mean(valid_error));

end

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

function [gen_loss,disc_loss] = computeloss(gw,dw,x,y,ws,nss,ovl)
    
    xh = inject_noise(x(end-ws+1:end,:,:));

    [dly,dlx_rwdt,dly_pl] = gen_predict(gw,x,xh,ws,nss,ovl);
    
    yes = disc_predict(dw,y.y);
    no = disc_predict(dw,permute(dly,[1 3 2]));

    disc_loss = -.5*mean(log(yes+eps) + log(1-no+eps),'all');

    rwdt_loss           = compute_reweight_detrend_loss(dlx_rwdt,y.x);       % compute reweight-detrend layer loss
    prediction_loss     = compute_prediction_loss(dly_pl,y.y_dt);            % compute prediction layer loss
    retrend_loss        = compute_retrend_loss(dly,y.y);                     % compute retrend layer loss

%     gen_loss.RDL   =  100*rwdt_loss;          
%     gen_loss.PL    = -.5*mean(log(no+eps),'all') + 100*prediction_loss;
%     gen_loss.IDL   = -.5*mean(log(no+eps),'all') + 100*retrend_loss;
%     gen_loss.E     = -.5*mean(log(no+eps),'all') + 100*mean([prediction_loss retrend_loss]);

    gen_loss = -.5*mean(log(no+eps),'all') + 100*mean([rwdt_loss prediction_loss retrend_loss]);   % combine layer & discriminator losses for gen loss
%             gen_loss = -.5*mean(log(no+eps),'all') + 100*mean(retrend_loss);
    
    function loss = compute_reweight_detrend_loss(dlx,x)
        % function for computing reweight-detrend-layer loss
        
%         loss = mean(abs(dlx-x),'all');

        loss = huber(dlx,x,'DataFormat','TCB');     % compute loss as smooth L1 between reweight-detrend layer output and reweighted-detrended reference sample
                                                                % this function's reduction method is summation by default (satisfies aml loss function)
    end

    function loss = compute_prediction_loss(dly,y)
        % function for computing prediction layer loss
        
%         loss = mean(abs(dly-squeeze(y)),'all');

        loss = huber(dly,squeeze(y),'DataFormat','TB');     % compute smooth L1 between prediction layer output and detrended output samples
    end

    function loss = compute_retrend_loss(dly,y)
        % function for computing retrend-layer loss
        
%         loss = mean(abs(dly-squeeze(y)),'all');
        loss = huber(dly,squeeze(y),'DataFormat','TB');     % compute smooth L1 between retrend layer output and expected output samples
    end
end

function [grad_gen,grad_disc] = modelgradients(gw,dw,xbatch,ybatch,ws,nss,ovl)

%     grad_gen.RDL    = dlgradient(gl.RDL,gw.RDL);       % compute reweight-detrend gradient
%     grad_gen.E      = dlgradient(gl.E,gw.E);           % compute encoder gradient
%     grad_gen.PL     = dlgradient(gl.PL,gw.PL);         % compute prediction layer gradient
%     grad_gen.IDL    = dlgradient(gl.IDL,gw.IDL);       % compute inverse detrend layer gradient

    [gl,dl] = computeloss(gw,dw,xbatch,ybatch,ws,nss,ovl);
    
    [grad_gen.RDL,grad_gen.E,grad_gen.PL,grad_gen.IDL] = dlgradient(gl,gw.RDL,gw.E,gw.PL,gw.IDL);
    grad_disc = dlgradient(dl,dw);                     % compute discriminator gradient
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

function [y,x_rwdt,y_pl] = gen_predict(gw,x,xh,ws,nss,ovl) % retrended y, reweighted detrended x, prediction layer y

    [x,ScaleFactor]                 = scalinglayer(x);                                          % Scale input data
    x_rwdt                          = gpudl(subsample_data(x,ws,nss,ovl),'');                   % Subsample data and load onto gpu
    [x_rwdt,ReweightDetrendInfo]    = reweightdetrendlayer(gw.RDL,x_rwdt);                      % Reweight and detrend subsampled data.
    dly_set                         = waveletlayer(x_rwdt);                                     % Wavelet transform dataset.
    dly_set                         = set_encodinglayer(gw.E.SEL,dly_set);                      % Encode input dataset. 
    y_pl                            = estimate_encodinglayer(gw.E.EEL,xh);                      % Encode previous estimate.
    y_pl                            = predictionlayer(gw.PL,dly_set,y_pl,ws);                   % Predict future system states from encoded data.
    y                               = inverse_detrendlayer(gw.IDL,y_pl,ReweightDetrendInfo);    % Re-apply trend. 

end

function [y] = disc_predict(dw,x)
    % function for forward pass on discriminator

    y = waveletlayer(permute(x,[1 3 2]));       % wavelet transform input data
    y = permute(y,[1 2 4 3]);                   % permute dly such that third dimension is singleton
    y = cat(3,real(y),imag(y));                 % split and concatenate real and imag values along third dimension
    
    for i = 1:length(dw.PL.blocks)                           % loop through blocks
        y = conv_maxpool(y,dw.PL.blocks{i},@leakyrelu);   % convolution x2 + maxpool blocks
    end
    
    y = squeeze(sigmoid(y));                            % remove singleton dimension
    
    function dly = conv_maxpool(dlx,layer,activation)   
        % function for convolution x2 + maxpool layer 
    
        dly = DeepNetwork.convlayer(dlx,layer.cn2d1,'DataFormat','SSCB');       % first 2D conv layer
        dly = activation(dly);                                                  % activation layer
    
        dly = DeepNetwork.convlayer(dly,layer.cn2d2,'DataFormat','SSCB');       % second 2D conv layer
        dly = activation(dly);                                                  % activation layer
        
        dly = DeepNetwork.batchnormlayer(dly,layer.bn1,'DataFormat','SSCB');    % batch norm layer               
    end
end


function y = subsample_data(x,ws,nss,ovl)
    % function for subsampling and rearranging scaled data
        
    batchsize = size(x,ndims(x));
    set = zeros([ws nss batchsize],'single');           % [ws nsubsamples batchsize] TCB
    
    locs = [1:ws];                    % Initialize locs 
    for i = 1:nss                     % Loop through subsamples
        set(:,i,:) = x(locs,:,:);     % Subsample x
        locs = locs + (ws-ovl);       % Increase locs
    end
    y = set;
end

function [y,rdi] = reweightdetrendlayer(lw,x)
    % layer for reweighting and detrending input data
        
    rdi = InputEncoder(x,lw.IE);                   % encode input data
    WeightCoeffs = ReweightDecoder(rdi,lw.RD);     % compute reweighing coefficients (outlier detection & removal)
    DetrendValues = DetrendDecoder(rdi,lw.DD);     % compute detrending values
    
    y = x.*WeightCoeffs + DetrendValues;       % reweight and detrend data
    
    function info = InputEncoder(dlx,layer)
        % layer for encoding input data into relevant information
    
        info = dlx;
        for i = 1:length(layer.blocks)                     % loop through layers
            info = encoder_block(info,layer.blocks{i},@tanh);    % encoder block
        end
    end
    
    function WeightCoeffs = ReweightDecoder(info,layer)
        % function for decoding encoder data into reweighting coefficients
        
        WeightCoeffs = info;
        for i = 1:length(layer.blocks)                                         % loop through layers
            WeightCoeffs = decoder_block(WeightCoeffs,layer.blocks{i},@tanh);  % decoder block tanh activation
        end
    end
    
    function DetrendValues = DetrendDecoder(info,layer)
        % function for decoding encoder data into values for detrending
    
        DetrendValues = info;
        for i = 1:length(layer.blocks)                                             % loop through layers
            DetrendValues = decoder_block(DetrendValues,layer.blocks{i},@tanh);    % decoder block tanh activation
        end
    end
    
    function dly = encoder_block(dlx,layer,activation)
        % function for encoder block
    
        dly = DeepNetwork.convlayer(dlx,layer.cn1d1,'DataFormat','SCB');               % first 1D convolution
        dly = activation(dly);                                                 % activation layer
    
        dly = DeepNetwork.convlayer(dly,layer.cn1d2,'DataFormat','SCB');               % second 1D convolution
        dly = activation(dly);                                                 % activation layer
    
        dly = DeepNetwork.batchnormlayer(dly,layer.bn1,'DataFormat','SCB');            % batch normalization layer
    
    end
    
    function dly = decoder_block(dlx,layer,activation)
        % function for decoder block
    
        dly = DeepNetwork.transposedconvlayer(dlx,layer.tcn1d1,'DataFormat','SCB',...      % transposed 1D convolution
            'Stride',[1]);       
        dly = activation(dly);                                                     % activation layer
    
        dly = DeepNetwork.batchnormlayer(dly,layer.bn1,'DataFormat','SCB');                % batch norm layer
    
    end
end

function y = set_encodinglayer(lw,x)
    % layer for encoding 

    %             dly = permute(dlx,[1 2 3 5 4]);     % permute data such that dimensions are SSSCB - they are STCB but TC -> SS and a new C dimension is added for 3d conv
    y = cat(3,real(x),imag(x));                                                   % separate wavelet layer output into real and complex components
    
    for i = 1:length(lw.blocks)                                     % loop through set encoding layer blocks
        y = set_encodingblock(y,lw.blocks{i},@tanh);            % set encoding blocks with tanh activation
    end
    
    %             dly = squeeze(dly);                                                               % remove one dimension
    
    function dly = set_encodingblock(dlx,layer,activation)
        % set sncoding blocks
        
        dly = DeepNetwork.groupedconvlayer(dlx,layer.gcn2d1,'DataFormat','SSCB');   % first grouped convolution
        dly = activation(dly);                                              % activation layer
    
        dly = DeepNetwork.groupedconvlayer(dly,layer.gcn2d2,'DataFormat','SSCB');   % second grouped convolution
        dly = activation(dly);                                              % activation layer
    
        dly = DeepNetwork.batchnormlayer(dly,layer.bn1,'DataFormat','SSCB');        % batch normalization layer
    
    end          
end

function y = estimate_encodinglayer(lw,x)

    % layer for encoding previous estimate 
        
    dlx = permute(x,[1 3 2]);         % Estimate is in the format TB, permute such that dimensions are TCB (1 Channel)
    y = waveletlayer(dlx);        % Wavelet transform the batch of estimates
    y = permute(y,[1 2 4 3]);       % permute dimensions such that they are SSCB
    y = cat(3,real(y),imag(y));   % split dly into real and complex parts
    
    for i = 1:length(lw.blocks)                                     % loop through blocks
        y = estimate_encoderblock(y,lw.blocks{i},@tanh);    % estimate encoder blocks with tanh activation
    end                                                    
    
    function dly = estimate_encoderblock(dlx,layer,activation)
        % function for estimate encoder block
        
        dly = DeepNetwork.convlayer(dlx,layer.cn2d1,'DataFormat','SSCB');       % first 2D convolution
        dly = activation(dly);                                                  % activation layer
    
        dly = DeepNetwork.convlayer(dly,layer.cn2d2,'DataFormat','SSCB');       % second 2D convolution
        dly = activation(dly);                                                  % activation layer
    
        dly = DeepNetwork.batchnormlayer(dly,layer.bn1,'DataFormat','SSCB');    % batch normalization layer
        
    end
end

function y = predictionlayer(lw,xs,xp,ws)
    % layer for predicting signal from encoded data
        
    y = xs;
    batchsize = size(y,4);
    
    for i = 1:length(lw.blocks2D)                                  % loop through blocks
        y = prediction_block2D(y,lw.blocks2D{i},@tanh);        % prediction decoder blocks
    end
    
    y = DeepNetwork.batchnormlayer(y+xp,lw.bn1,'DataFormat','SSCB');  % combine set and estimate data
          
    y = reshape(y,ws,[],batchsize);                                % reshape output 
    
    for i = 1:length(lw.blocks1D)                           % loop through 1D blocks
        y = prediction_block1D(y,lw.blocks1D{i},@tanh); % 1D block
    end
    
    y = squeeze(y); % remove singleton dimension
    
    function dly = prediction_block2D(dlx,layer,activation)
        % function for 2D prediction layer block
    
        dly = DeepNetwork.transposedconvlayer(dlx,layer.tcn2d1,'DataFormat','SSCB');    % transposed 2D convolution layer
        dly = activation(dly);                                                  % activation function
    
        dly = DeepNetwork.batchnormlayer(dly,layer.bn1,'DataFormat','SSCB');            % batch normalization layer  
    end
    
    function dly = prediction_block1D(dlx,layer,activation)
        % function for 1D prediction layer block
    
        dly = DeepNetwork.convlayer(dlx,layer.cn1d1,'DataFormat','SCB',...          % 1D convolution layer
            'Padding','same','PaddingValue','symmetric-include-edge');            
        dly = activation(dly);                                              % activation function
        dly = DeepNetwork.batchnormlayer(dly,layer.bn1,'DataFormat','SCB');         % batch norm layer
    end
end

function y = inverse_detrendlayer(lw,x,rdi)

    % layer which predicts and re-applies trend to prediction
       
    y = rdi;                                            % set dly as detrend info from reweight-detrend layer
        
    for i = 1:length(lw.blocks)                            % loop through blocks
        y = idetrend_block(y,lw.blocks{i},@tanh);       % inverse-detrend block
    end
    
    y = (permute(x,[1 3 2])+y);
    y = DeepNetwork.batchnormlayer(y,lw.bn1,'DataFormat','SCB');      % retrend dlx and scale
    y = squeeze(y);                                                   % remove singleton dimension
    
    function dly = idetrend_block(dlx,layer,activation)
        % function for detrend block
    
        dly = DeepNetwork.transposedconvlayer(dlx,layer.tcn1d1,'DataFormat','SCB');     % transposed 1D convolution
        dly = activation(dly);                                                          % activation function
    
        dly = DeepNetwork.batchnormlayer(dly,layer.bn1,'DataFormat','SCB');             % batch norm layer
    end
end
















































