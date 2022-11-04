%{
        Class for Generator

        Jashua Luna
        October 2022

        Generator object (makes price predictions).
%}

classdef generator < DeepNetwork
    properties
        previous_estimate
        data
    end

    methods
        function obj = generator(layersizes,WindowSize,Overlap,nSubsamples,BatchSize)
             % constructor

             if isempty(layersizes)                             % if layersizes are unspecified
                 layersizes = generator.layersizes_generator;   % use default sizes
             end

             obj.info.WindowSize    = WindowSize;       % Sample windowSize
             obj.info.Overlap       = Overlap;          % Subsample overlap
             obj.info.nSubsamples   = nSubsamples;      % No. subsamples     
             obj.info.BatchSize     = BatchSize;        % Batch Size

             obj.data               = get_data;         % read historical price data from file

%              [obj.weights.SL,obj.info.SL]       = obj.init_scalinglayer(layersizes.SL);               % Initialize scaling layer.
             [obj.weights.RDL,obj.info.RDL]     = obj.init_reweightdetrendlayer(layersizes.RDL);        % Initialize reweight-detrend layer.
             [~,obj.info.WTL]                   = obj.init_waveletlayer(layersizes.WTL);                % Initialize wavelet layer.
             [obj.weights.SEL,obj.info.SEL]     = obj.init_set_encodinglayer(layersizes.SEL);           % Initialize set-encoder.
             [obj.weights.EEL,obj.info.EEL]     = obj.init_estimate_encodinglayer(layersizes.EEL);      % Initialize estimate-encoder.
             [obj.weights.PL,obj.info.PL]       = obj.init_predictionlayer(layersizes.PL);              % Initialize prediction layer.
             [obj.weights.IDL,obj.info.IDL]     = obj.init_inversedetrendlayer(layersizes.IDL);         % Initialize inverse-detrend layer.
%              [obj.weights.ISL,obj.info.ISL]     = obj.init_inversescalinglayer(layersizes.ISL);         % Initialize inverse-scaling layer.

        end
        
        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %
        %       Network & Layer Functions

        function [xdata,ydata] = get_trainingdata(obj,nSamples)
            % function for generating training samples

            

        end

        function [xbatch,ybatch] = get_batch(obj,xdata,ydata)
            % function for converting cell arrays of training data to dl,nD arrays on  gpu



        end
        
        function [gen_loss,disc_loss] = compute_losses(obj,xbatch,outputs,disc)
            % function for computing generator and discriminator loss        

            [dly,dlx_rwdt,dly_pl] = obj.predict(xbatch,xhatbatch);     % feed forward through generator [re-trended output, de-trended & re-weighed input, prediction layer output] 

            yes = disc.predict(outputs.y_dt,inject_noise(outputs.y));       % determine discriminator output on real samples
            no  = disc.predict(dly);                                        % determine discriminator output on predicted samples

            rwdt_loss           = compute_reweight_detrend_loss(dlx_rwdt,outputs.x);       % compute reweight-detrend layer loss
            prediction_loss     = compute_prediction_loss(dly_pl,outputs.y_dt);            % compute prediction layer loss
            retrend_loss        = compute_retrend_loss(dly,outputs.y);                     % compute retrend layer loss

            disc_loss = -.5*mean(log(yes+eps) + log(1-no+eps),'all');   % discriminator loss

            gen_loss = -.5*mean(log(no+eps),'all') + 5*rwdt_loss + 50*prediction_loss + 100*retrend_loss;   % combine layer & discriminator losses for gen loss
            
            function loss = compute_reweight_detrend_loss(dlx,x)
                % function for computing reweight-detrend-layer loss
                
                loss = huber(dlx,dlarray(x,''),'DataFormat','TCB');     % compute loss as smooth L1 between reweight-detrend layer output and reweighted-detrended reference sample
                                                                        % this function's reduction method is summation by default (satisfies aml loss function)
            end

            function loss = compute_prediction_loss(dly,y)
                % function for computing prediction layer loss

                loss = huber(dly,dlarray(y,''),'DataFormat','TB');      % compute smooth L1 between prediction layer output and detrended output samples
            end

            function loss = compute_retrend_loss(dly,y)
                % function for computing retrend-layer loss
                
                loss = huber(dly,dlarray(y,''),'DataFormat','TB');      % compute smooth L1 between retrend layer output and expected output samples
            end
        end

        function y = reweight_detrend(obj,x)
            % function for generating reweighted-detrended samples 
            % where x is a two dimensional array TB 

            x = huber_reweight(obj.subsample(x));       % subsample and reweight x 
            y = obj.waveletlayer(x);                    % obtain the waveletlayer output of x
            y = inverse_waveletlayer(y);                % inverse wavelet layer y
            
            function xbatch = huber_reweight(xbatch)
                % function for reweighting xbatch with huber weigh function
                % xbatch is in format TB

                sf  = mad(xbatch,1,1)/0.6745;               % compute scale factor as median absolute deviation divided by 0.6745
                r   = abs(xbatch - median(xbatch,1))/sf;    % compute residual as abs delta x - median(x) divided by sf

                w = huber_weight(r,1,547);      % get huber weights
                xbatch = w.*xbatch;             % reweight samples
            end

            function weights = huber_weight(x,c)
                % function for determining huber weights

                weights = zeros(size(x));
                for i = 1:size(x,2)
                    gtc = x(:,i)>= c(i);                   % locations of x values greater than or equal to c
                    weights(gtc,i) = 1;                    % weights at locs = 1
                    weights(~gtc,i) = c(i)/x(~gtc,i);      % otherwise weights = c/x
                end                 
            end
        end

        function [dly,varargout] = predict(obj,x,varargin)
            % feed forward function

            if obj.IsTraining                       % If network is training, xhat is supplied as training data.
                xhat = varargin{1};
            else                                    % Otherwise, xhat is stored in the network. 
                xhat = obj.previous_estimate;
            end
            
            [x,ScaleFactor]                     = obj.scalinglayer(x);                                              % Scale input data 
            dlx                                 = obj.subsample_data(x);                                            % Subsample data
            [dlx,ReweightDetrendInfo]           = obj.reweightdetrendlayer(dlx);                                    % Reweight and detrend subsampled data.
            dly_set                             = obj.waveletlayer(dlx);                                            % Wavelet transform dataset.
            dly_set                             = obj.set_encodinglayer(dly_set);                                   % Encode input dataset. 
            dly_estimate                        = obj.estimate_encodinglayer(xhat);                                 % Encode previous estimate.
            dly_estimate                        = obj.predictionlayer(dly_set,dly_estimate);                        % Predict future system states from encoded data.
            obj.debug_info("Reweight-Detrend Info Size: ",ReweightDetrendInfo);                                     % display reweight-detrend info size when in debug
            dly1                                = obj.inverse_detrendlayer(dly_estimate,ReweightDetrendInfo);       % Re-apply trend. 
%             dly2                                = obj.inverse_scalinglayer(dly1,FineTuneCoeff,ScaleFactor);         % Re-scale predicted values. 
            
            dly = dly1;

            if ~obj.IsTraining                                  % If network is not training store prediction and return self
                obj.previous_estimate = dly_estimate;           % It would be more efficient to have two separate functions for training and use
                varargout{1} = obj;
            else                                                % Otherwise, return all values relevant to the loss function.
%                 varargout{1} = x;                               % Scaled data.
                varargout{1} = dlx;                             % Reweighted and Detrended Data.
                varargout{2} = dly_estimate;                    % Un-scaled prediction.
%                 varargout{3} = dly1;                            % Re-trended prediction.
%                 varargout{5} = dly2;                          % Re-scaled, re-trended prediction.
            end
        end

        function dlx = subsample_data(obj,x)
            % function for subsampling and rearranging scaled data
            persistent set

            obj.debug_info("<strong>Subsampling Layer</strong>",[]);        % display active layer's name while in debug
            
            if isempty(set)                                                 % if persistent variable is empty initialize set as zeros
                set = gpudl(zeros([obj.info.WindowSize obj.info.nSubsamples obj.info.BatchSize],obj.DataType),'');           % [ws nsubsamples batchsize]
                %       [windowsize nsumbsamples batchsize] TCB
            end
            
            locs = [1:obj.info.WindowSize];                                 % Initialize locs 
            for i = 1:obj.info.nSubsamples                                  % Loop through subsamples
                set(:,i,:) = x(locs,:,:);                                   % Subsample x
                locs = locs + (obj.info.WindowSize-obj.info.Overlap);       % Increase locs
            end
            dlx = set;
            obj.debug_info("Reshaped Data Sizes: ",dlx);                    % display output's dimensions while in debug    [TCB]
        end

        function [dly,ScaleFactor] = scalinglayer(obj,x)
            % layer for scaling and subsampling input data

            obj.debug_info("<strong>Scaling Layer</strong>",[]);                    % display active layer's name while in debug

            ScaleFactor = range(abs(x),1);                                          % scale input data by the range of its absolute values
            dly = gpudl(x./ScaleFactor,'');                                         %
            obj.debug_info("Scaled Data Sizes: ",dly);                              % display output's dimensions while in debug
        end

        function [dly,info] = reweightdetrendlayer(obj,dlx)
            % layer for reweighting and detrending input data

            obj.debug_info("<strong>Reweight and Detrend Layer</strong>",[]);       % display active layer's name while in debug

            info = InputEncoder(dlx,obj.weights.RDL.IE);                            % encode input data
            WeightCoeffs = ReweightDecoder(info,obj.weights.RDL.RD);                % compute reweighing coefficients (outlier detection & removal)
            DetrendValues = DetrendDecoder(info,obj.weights.RDL.DD);                % compute detrending values

            dly = dlx.*WeightCoeffs + DetrendValues;                                % reweight and detrend data

            function info = InputEncoder(dlx,layer)
                % layer for encoding input data into relevant information

                obj.debug_info("<strong>Input Encoder</strong>",[]);                            % display active layer's name while in debug
                
                info = dlx;
                for i = 1:obj.info.RDL.IE.nBlocks                                               % loop through layers
                    info = encoder_block(obj.dropout(info),layer.blocks{i},@relu);              % encoder block

                    debug_message = append("Output size after ",num2str(i)," iterations ");     % debug message
                    obj.debug_info(debug_message,info);                                         % display layer output size 
                end
            end

            function WeightCoeffs = ReweightDecoder(info,layer)
                % function for decoding encoder data into reweighting coefficients
                
                obj.debug_info("<strong>Reweight Decoder</strong>",[]);                             % display active layer's name while in debug
                
                WeightCoeffs = info;
                for i = 1:obj.info.RDL.RD.nBlocks                                                   % loop through layers
                    WeightCoeffs = decoder_block(obj.dropout(WeightCoeffs),layer.blocks{i},@tanh);  % decoder block tanh activation

                    debug_message = append("Output size after ",num2str(i)," iterations ");         % debug message
                    obj.debug_info(debug_message,WeightCoeffs);                                     % display layer output size 
                end
            end

            function DetrendValues = DetrendDecoder(info,layer)
                % function for decoding encoder data into values for detrending

                obj.debug_info("<strong>Detrend Decoder</strong>",[]);                                  % display active layer's name while in debug


                DetrendValues = info;
                for i = 1:obj.info.RDL.DD.nBlocks                                                       % loop through layers
                    DetrendValues = decoder_block(DetrendValues,layer.blocks{i},@tanh);                    % decoder block tanh activation

                    debug_message = append("Output size after ",num2str(i)," iterations ");             % debug message
                    obj.debug_info(debug_message,DetrendValues);                                        % display layer output size 
                end
            end

            function dly = encoder_block(dlx,layer,activation)
                % function for encoder block

                dly = obj.convlayer(dlx,layer.cn1d1,'DataFormat','SCB');               % first 1D convolution
                dly = activation(dly);                                                 % activation layer

                dly = obj.convlayer(dly,layer.cn1d2,'DataFormat','SCB');               % second 1D convolution
                dly = activation(dly);                                                 % activation layer

                dly = obj.maxpoollayer(dly,[9],'Stride',[1],'DataFormat','SCB');       % max pooling layer
                dly = obj.batchnormlayer(dly,layer.bn1,'DataFormat','SCB');            % batch normalization layer


            end

            function dly = decoder_block(dlx,layer,activation)
                % function for decoder block

                dly = obj.transposedconvlayer(dlx,layer.tcn1d1,'DataFormat','SCB',...      % transposed 1D convolution
                    'Stride',[1]);       
                dly = activation(dly);                                                     % activation layer

                dly = obj.batchnormlayer(dly,layer.bn1,'DataFormat','SCB');                % batch norm layer

            end
        end

        function [dly] = waveletlayer(obj,dlx)
            % layer to wavelet transform input data

            obj.debug_info("<strong>Wavelet Layer</strong>",[]);                                    % display active layer's name while in debug

            dly = dlcwt(dlx,obj.info.WTL.psi_fvec,obj.info.WTL.filter_idx,'DataFormat','TCB');      % Wavelet transform subsampled sets
            dly = permute(dly,[1 4 2 3]);                                                           % dlcwt output is SCBT where S is filter dilation, permute to STCB

            obj.debug_info("Output size after Wavelet Layer: ",dly);                                % display layer output size
        end

        function [dly] = set_encodinglayer(obj,dlx)
            % layer for encoding 

            obj.debug_info("<strong>Set Encoding Layer</strong>",[]);                           % display active layer's name while in debug
            
            dly = permute(dlx,[1 2 3 5 4]);     % permute data such that dimensions are SSSCB - they are STCB but TC -> SS and a new C dimension is added for 3d conv
            dly = cat(4,real(dly),imag(dly));                                                   % separate wavelet layer output into real and complex components
            obj.debug_info("Output size after permute operation and concatenation: ",dly);      % display layer output size

            for i = 1:obj.info.SEL.nBlocks                                                      % loop through set encoding layer blocks
                dly = set_encodingblock(obj.dropout(dly),obj.weights.SEL.blocks{i},@relu);      % set encoding blocks with relu activation

                debug_message = append("Output size after ",num2str(i)," iterations ");         % debug message
                obj.debug_info(debug_message,dly);                                              % display layer output size 
            end

            dly = squeeze(dly);                                                                 % remove one dimension

            function dly = set_encodingblock(dlx,layer,activation)
                % set sncoding blocks
                
                dly = obj.convlayer(dlx,layer.cn3d1,'DataFormat','SSSCB');                      % first 3D convolution
                dly = activation(dly);                                                          % activation layer

                dly = obj.convlayer(dly,layer.cn3d2,'DataFormat','SSSCB');                      % second 3D convolution
                dly = activation(dly);                                                          % activation layer

                dly = obj.maxpoollayer(dly,[4 4 4],'Stride',[1 1 1],'DataFormat','SSSCB');      % maximum pooling layer

                dly = obj.batchnormlayer(dly,layer.bn1,'DataFormat','SSSCB');                   % batch normalization layer
   
            end            
        end

        function [dly] = estimate_encodinglayer(obj,dlx)
            % layer for encoding previous estimate 

            obj.debug_info("<strong>Estimate Encoding Layer</strong>",[]);                           % display active layer's name while in debug

            dlx = permute(dlx,[1 3 2]);         % Estimate is in the format TB, permute such that dimensions are TCB (1 Channel)
            dly = obj.waveletlayer(dlx);        % Wavelet transform the batch of estimates
            dly = cat(3,real(dly),imag(dly));   % split dly into real and complex parts

            for i = 1:obj.info.EEL.nBlocks                                                          % loop through blocks
                dly = estimate_encoderblock(obj.dropout(dly),obj.weights.EEL.blocks{i},@relu);      % estimate encoder blocks with relu activation

                debug_message = append("Output size after ",num2str(i)," iterations ");     % debug message
                obj.debug_info(debug_message,dly);                                          % display layer output size 
            end                                                    

            function dly = estimate_encoderblock(dlx,layer,activation)
                % function for estimate encoder block
                
                dly = obj.convlayer(dlx,layer.cn2d1,'DataFormat','SSCB');                   % first 2D convolution
                dly = activation(dly);                                                      % activation layer

                dly = obj.convlayer(dly,layer.cn2d2,'DataFormat','SSCB');                   % second 2D convolution
                dly = activation(dly);                                                      % activation layer

                dly = obj.maxpoollayer(dly,[4 8],'Stride',[1 1],'DataFormat','SSCB');       % max pooling layer

                dly = obj.batchnormlayer(dly,layer.bn1,'DataFormat','SSCB');                % batch normalization layer
                
            end
        end

        function [dly,obj] = predictionlayer(obj,dlx_set,dlx_estimate)
            % layer for predicting signal from encoded data

            obj.debug_info("<strong>Prediction Layer</strong>",[]);                           % display active layer's name while in debug

            dly = dlx_set;
            
            for i = 1:obj.info.PL.nBlocks2D                                                     % loop through blocks
                dly = prediction_block2D(obj.dropout(dly),obj.weights.PL.blocks2D{i},@tanh);    % prediction decoder blocks
                
                debug_message = append("Output size after ",num2str(i)," iterations ");         % debug message
                obj.debug_info(debug_message,dly);                                              % display layer output size 
            end

            dly = obj.batchnormlayer(obj.dropout(dly+dlx_estimate),obj.weights.PL.bn1,...       % combine set and estimate data
                'DataFormat','SSCB');  
            dly = reshape(dly,obj.info.WindowSize,[],obj.info.BatchSize);                       % reshape output 
            obj.debug_info("Output size after reshape : ",dly);                                 % display output size

            for i = 1:obj.info.PL.nBlocks1D                                                     % loop through 1D blocks
                dly = prediction_block1D(obj.dropout(dly),obj.weights.PL.blocks1D{i},@tanh);    % 1D block
                
                debug_message = append("Output size after ",num2str(i)," iterations ");     % debug message
                obj.debug_info(debug_message,dly);                                          % display layer output size 
            end

            dly = squeeze(dly); % remove singleton dimension
            obj.debug_info("output size after prediction layer: ",dly);

            function dly = prediction_block2D(dlx,layer,activation)
                % function for 2D prediction layer block

                dly = obj.transposedconvlayer(dlx,layer.tcn2d1,'DataFormat','SSCB');    % transposed 2D convolution layer
                dly = activation(dly);                                                  % activation function

                dly = obj.batchnormlayer(dly,layer.bn1,'DataFormat','SSCB');            % batch normalization layer  
            end

            function dly = prediction_block1D(dlx,layer,activation)
                % function for 1D prediction layer block

                dly = obj.convlayer(dlx,layer.cn1d1,'DataFormat','SCB',...          % 1D convolution layer
                    'Padding','same','PaddingValue','symmetric-include-edge');            
                dly = activation(dly);                                              % activation function

                dly = obj.batchnormlayer(dly,layer.bn1,'DataFormat','SCB');         % batch norm layer
            end
        end

        function [dly] = inverse_detrendlayer(obj,dlx,DetrendInfo)
            % layer which predicts and re-applies trend to prediction

            obj.debug_info("<strong>Inverse Detrend Layer</strong>",[]);                    % display active layer's name while in debug
            
            dly = DetrendInfo;                                                              % set dly as detrend info from reweight-detrend layer
                
            for i = 1:obj.info.IDL.nBlocks                                                   % loop through blocks
                dly = idetrend_block(obj.dropout(dly),obj.weights.IDL.blocks{i},@tanh);      % inverse-detrend block

                debug_message = append("Output size after ",num2str(i)," iterations ");     % debug message
                obj.debug_info(debug_message,dly);                                          % display layer output size 
            end

            dly = squeeze(dlx + dly);            % retrend dlx

            function dly = idetrend_block(dlx,layer,activation)
                % function for detrend block

                dly = obj.transposedconvlayer(dlx,layer.tcn1d1,'DataFormat','SCB');         % transposed 1D convolution
                dly = activation(dly);                                                      % activation function

                dly = obj.batchnormlayer(dly,layer.bn1,'DataFormat','SCB');                 % batch norm layer
            end
        end

%         function [dly] = inverse_scalinglayer(obj,dlx,FineTuneCoeff,ScaleFactor)
%                   omitted for now because it is not necessary to re-sclae
%                   data at this stage
%         end

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %
        %           Layer Initialization Functions

%         function [layer,info] = init_scalinglayer(obj,layersizes)
%             
%         end

        function [layer,info] = init_reweightdetrendlayer(obj,layersizes)
            % function for initializing reweight-detrend layer

            info.IE.nBlocks         = .5*size(layersizes.IE,1);                         % determine number of input encoder blocks
            assert(rem(info.IE.nBlocks,1)==0,"rows in encoder layersizes must be a factor of two");

            info.RD.nBlocks         = size(layersizes.RD,1);                            % determine number of reweight decoder blocks
            info.DD.nBlocks         = size(layersizes.DD,1);                            % determine number of detrend decoder blocks

            layer.IE = init_inputencoder(layersizes.IE);                                % initialize input encoder
            layer.RD = init_reweightdecoder(layersizes.RD);                             % initialize reweight decoder
            layer.DD = init_detrenddecoder(layersizes.DD);                              % initialize detrend decoder

            function layer = init_inputencoder(layersizes)
                % function for initializing input encoder

                layer.blocks = cell(info.IE.nBlocks,1);                                 % create cell array for blocks
        
                for i = 1:info.IE.nBlocks                                               % loop through blocks
                    layer.blocks{i} = init_encoder_block(layersizes(2*i-1:2*i,:));      % initialize layerblocks
                end
            end

            function layer = init_reweightdecoder(layersizes)
                % function for initializing reweight decoder

                layer.blocks = cell(info.RD.nBlocks,1);                                 % create cell array for blocks

                for i = 1:info.RD.nBlocks                                               % loop through blocks
                    layer.blocks{i} = init_decoder_block(layersizes(i,:));              % initialize layerblocks
                end
            end

            function layer = init_detrenddecoder(layersizes)
                % function for initializing detrend decoder

                layer.blocks = cell(info.DD.nBlocks,1);                                 % create cell array for blocks
            
                for i = 1:info.DD.nBlocks                                               % loop through blocks
                    layer.blocks{i} = init_decoder_block(layersizes(i,:));              % initialize layerblocks
                end
            end

            function layer = init_encoder_block(layersizes)
                % initialize encoder blocks
                layer.cn1d1 = obj.init_convlayer(layersizes(1,1),layersizes(1,2),layersizes(1,3),obj.DataType);    % init first 1D convolution
                layer.cn1d2 = obj.init_convlayer(layersizes(2,1),layersizes(2,2),layersizes(2,3),obj.DataType);    % init first 1D convolution
                layer.bn1   = obj.init_batchnormlayer(layersizes(2,3),obj.DataType);                               % init batch norm layer
            end

            function layer = init_decoder_block(layersizes)
                % initialize decoder block

                layer.tcn1d1    = obj.init_transposedconvlayer(layersizes(1),layersizes(2),layersizes(3),obj.DataType);     % init transposed convolution
                layer.bn1       = obj.init_batchnormlayer(layersizes(3),obj.DataType);                                      % init batchnorm layer
            end
        end

        function [layer,info] = init_waveletlayer(obj,layersizes,varargin)
            % function for initializing wavelet layer

            wavelet = cwtfilterbank('SignalLength',obj.info.WindowSize,'Boundary','periodic',varargin{:});  % create a filter wavlet bank
            [info.psi_fvec,info.filter_idx] = cwtfilters2array(wavelet);                                    % store filter bank in network info

            layer = []; % empty array for semantics

        end

        function [layer,info] = init_set_encodinglayer(obj,layersizes)
            % function for initializing set encoding layer

            info.nBlocks = .5*size(layersizes,1);                                                           % number of blocks is half the rows of layersizes
            assert(rem(info.nBlocks,1)==0,"Number of rows in layersizes must be a factor of two\n");        % ensure the number of rows is even
            layer.blocks = cell(info.nBlocks,1);                                                            % create cell array for blocks

            for i = 1:info.nBlocks                                                                          % loop through blocks
                layer.blocks{i} = init_set_encodingblock(layersizes(2*i-1:2*i,:));                          % initialize blocks
            end

            function block = init_set_encodingblock(ls)            
                % function for initializing set encoder blocks

                block.cn3d1 = obj.init_convlayer(ls(1,1:3),ls(1,4),ls(1,5),obj.DataType);                   % init first 3D conv layer
                block.cn3d2 = obj.init_convlayer(ls(2,1:3),ls(2,4),ls(2,5),obj.DataType);                   % init second 3D conv layer
                block.bn1   = obj.init_batchnormlayer(ls(2,5),obj.DataType);                                % init batchnorm layer
                           
            end
        end

        function [layer,info] = init_estimate_encodinglayer(obj,layersizes)
            % function for initializing estimate encoder blocks

            info.nBlocks = .5*size(layersizes,1);                                                           % determine number of blocks
            assert(rem(info.nBlocks,1)==0,"Number of rows in layersizes must be a factor of two\n");        % ensure the number of rows is even
            layer.blocks = cell(info.nBlocks,1);                                                            % init blocks as cell function        

            for i = 1:info.nBlocks                                                                          % loop through blocks
                layer.blocks{i} = init_estimate_encoderblock(layersizes(2*i-1:2*i,:));                      % init blocks
            end

            function block = init_estimate_encoderblock(ls)
                % function for initializing estimate encoder block

                block.cn2d1 = obj.init_convlayer(ls(1,1:2),ls(1,3),ls(1,4),obj.DataType);           % init first 2D conv layer
                block.cn2d2 = obj.init_convlayer(ls(2,1:2),ls(2,3),ls(2,4),obj.DataType);           % init second 2D conv layer
                block.bn1   = obj.init_batchnormlayer(ls(2,4),obj.DataType);                        % init batchnorm layer
            end
        end

        function [layer,info] = init_predictionlayer(obj,layersizes)
            % function for initializing prediction layer
            
            info.nBlocks2D = size(layersizes.w2D,1);                                        % determine number of two dimensional convolution layers
            info.nBlocks1D = size(layersizes.w1D,1);                                        % determine number of prediction blocks

            layer.blocks2D = cell(info.nBlocks2D,1);                                        % init 2D conv layers
            layer.blocks1D = cell(info.nBlocks1D,1);                                        % init 1D conv layers

            for i = 1:info.nBlocks2D                                                        % loop through 2D blocks
                layer.blocks2D{i} = init_predictionblock2D(layersizes.w2D(i,:));            % loop prediction block
            end
                
            layer.bn1 = obj.init_batchnormlayer(layersizes.w2D(end,4),obj.DataType);        % init batchnorm layer for addition 

            for i = 1:info.nBlocks1D                                                        % loop through 1D blocks
                layer.blocks1D{i} = init_predictionblock1D(layersizes.w1D(i,:));            % init 1D prediction blocks
            end

            function block = init_predictionblock2D(ls)
                % function for initializing 2D prediction block

                block.tcn2d1    = obj.init_transposedconvlayer(ls(1:2),ls(3),ls(4),obj.DataType);   % init transposed conv layer
                block.bn1       = obj.init_batchnormlayer(ls(4),obj.DataType);                      % init batchnorm layer

            end

            function block = init_predictionblock1D(ls)
                % function for initializing 1D prediction block

                block.cn1d1     = obj.init_convlayer(ls(1),ls(2),ls(3),obj.DataType);   % init 2D conv
                block.bn1       = obj.init_batchnormlayer(ls(3),obj.DataType);          % init batchnorm layer
            end
        end

        function [layer,info] = init_inversedetrendlayer(obj,layersizes)
            % function for initializing inverse detrend layer

            info.nBlocks = size(layersizes,1);                                      % determine number of blocks
            layer.blocks = cell(info.nBlocks,1);                                    % init cell array for blocks
                
            for i = 1:info.nBlocks                                                  % loop through blocks
                layer.blocks{i} = init_inversedetrendblock(layersizes(i,:));        % init blokcs
            end

            function layer = init_inversedetrendblock(ls)
                % function for initializing inverse detrend block

                layer.tcn1d1    = obj.init_transposedconvlayer(ls(1),ls(2),ls(3),obj.DataType); % transposed 1D convolution layer
                layer.bn1       = obj.init_batchnormlayer(ls(3),obj.DataType);                  % batchnorm layer
            end
        end

%         function [layer,info] = init_inversescalinglayer(obj,layersizes)
%                         layer = [];
%             info = [];
% 
%         end        

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %       Other Functions

        function obj = weightless_copy(obj)
            % function for copying generator object without weights

            obj.weights = [];       % empty weights
        end
    end

    methods (Static)
            
        function layersizes = layersizes_generator
            % function for default generator sizes

            layersizes.RDL  = reweightdetrend_layersizes;       % reweight-detrend layer
            layersizes.WTL  = [];                               % wavelet layer
            layersizes.SEL  = setencoder_layersizes;            % set encoding layer
            layersizes.EEL  = estimateencoder_layersizes;       % estimate encoding layer
            layersizes.PL   = predictionlayer_layersizes;       % prediction layer
            layersizes.IDL  = inversedetrend_layersizes;        % inverse detrend layer


            function layersizes = reweightdetrend_layersizes
                % function for reweight detrend layer sizes
                layersizes.IE   = [     % Reweight-Detrend Layer -> Input Encoder
                    8   64  64;
                    8   64  128;
                
                    4   128 256;
                    4   256  512;
                
                    4   512 512;
                    4   512 512; 
                    ];   

                layersizes.RD   = [     % Reweight-Detrend Layer -> Reweight Decoder
                    12   512     256;
                    12   256     128;
                    12   128     64;
                    11   64      64;
                    8    64      64;
                    ];  

                layersizes.DD   = [     % Reweight-Detrend Layer -> Detrend Decoder
                    12   512     256;
                    12   256     128;
                    12   128     64;
                    11   64      64;    
                    8    64      64;
                    ];   
            end

            function layersizes = setencoder_layersizes
                % function for set encoder layer sizes

                layersizes  = [             % Set Encoding Layer
                    4   9   7   2   16;
                    4   9   7   16  32;
                
                    4   9   7   32  64;
                    4   9   7   64  128;
                
                    4   7   5   128  256;
                    4   7   5   256  512;
                
                    3   5   3   512 512;
                    2   5   3   512 512;
                    ];  
            end

            function layersizes = estimateencoder_layersizes
                % function for estimate encoder layer sizes

                layersizes  = [         % Estimate Encoding Layer
                    3   16   2   32;
                    3   16   32  64;
                
                    3   9   64  128;
                    3   9   128  256;
                    ];   
            end

            function layersizes = predictionlayer_layersizes
                % function for prediction layer sizes

                layersizes.w2D   = [         % Prediction layer 2D component
                    3   3   512    512;
                    3   3   512     256;
                    ];
                
                layersizes.w1D   = [         % Prediction Layer 1D component
                    8   1280    1024;
                    8   1024   512;
                    8   512     256;     
                    4   256     128;
                    4   128     64;
                    4   64      32;
                    2   32      16;
                    2   16      8;
                    2   8       4;
                    1   4       1;
                    ];   
            end

            function layersizes = inversedetrend_layersizes
                % function for inverse-detrend layer sizes

                layersizes  = [     
                    8   512 256;
                    8   256 128;
                    8   128 64;
                    8   64  32;
                    8   32  16;
                    8   16  8;
                    6   8   4;
                    4   4   1;
                    ];   
            end
        end
    end
end







































































