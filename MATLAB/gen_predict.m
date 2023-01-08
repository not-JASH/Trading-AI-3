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