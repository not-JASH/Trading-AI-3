classdef generator < DeepNetwork
    properties
        previous_estimate
    end

    methods
        function obj = generator(layersizes,WindowSize,Overlap,nSubsamples)
             % constructor

             obj.info.WindowSize    = WindowSize;       % Sample windowSize
             obj.info.Overlap       = Overlap;          % Subsample overlap
             obj.info.nSubsamples   = nSubsamples;      % No. subsamples                

%              [obj.weights.SL,obj.info.SL]       = obj.init_scalinglayer(layersizes.SL);               % Initialize scaling layer.
             [obj.weights.RDL,obj.info.RDL]     = obj.init_reweightdetrendlayer(layersizes.RDL);        % Initialize reweight-detrend layer.
             [~,obj.info.WTL]                   = obj.init_waveletlayer(layersizes.WTL);                % Initialize wavelet layer.
             [obj.weights.SEL,obj.info.SEL]     = obj.init_set_encodinglayer(layersizes.SEL);           % Initialize set-encoder.
             [obj.weights.EEL,obj.info.EEL]     = obj.init_estimate_encodinglayer(layersizes.EEL);      % Initialize estimate-encoder.
             [obj.weights.PL,obj.info.PL]       = obj.init_predictionlayer(layersizes.PL);              % Initialize prediction layer.
             [obj.weights.IDL,obj.info.IDL]     = obj.init_inversedetrendlayer(layersizes.IDL);         % Initialize inverse-detrend layer.
             [obj.weights.ISL,obj.info.ISL]     = obj.init_inversescalinglayer(layersizes.ISL);         % Initialize inverse-scaling layer.

        end
        
        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %
        %       Network & Layer Functions


        function [dly,varargout] = predict(obj,x,varargin)
            % feed forward function

            if obj.IsTraining                       % If network is training, xhat is supplied as training data.
                xhat = varargin{1};
            else                                    % Otherwise, xhat is stored in the network. 
                xhat = obj.previous_estimate;
            end
            
            [x,ScaleFactor]                     = obj.scalinglayer(x);                                              % Scale input data 
            dlx                                 = subsample_data(x);                                                % Subsample data
            [dlx,ReweightDetrendInfo]           = obj.reweightdetrendlayer(dlx);                                    % Reweight and detrend subsampled data.
            dly_set                             = obj.waveletLayer(dlx);                                            % Wavelet transform dataset.
            dly_set                             = obj.set_encodinglayer(dly_set);                                   % Encode input dataset. 
            dly_estimate                        = obj.estimate_encodinglayer(xhat);                                 % Encode previous estimate.
            dly_estimate                        = obj.predictionlayer(dly_set,dly_estimate);                        % Predict future system states from encoded data.
            dly1                                = obj.inverse_detrendlayer(dly_estimate,ReweightDetrendInfo);       % Re-apply trend. 
%             dly2                                = obj.inverse_scalinglayer(dly1,FineTuneCoeff,ScaleFactor);         % Re-scale predicted values. 
            
            dly = dly2;

            if ~obj.IsTraining                                  % If network is not training store prediction and return self
                obj.previous_estimate = dly_estimate;           % It would be more efficient to have two separate functions for training and use
                varargout{1} = obj;
            else                                                % Otherwise, return all values relevant to the loss function.
                varargout{1} = x;                               % Scaled data.
                varargout{2} = dlx2;                            % Reweighted and Detrended Data.
                varargout{3} = dly_estimate;                    % Un-scaled prediction.
                varargout{4} = dly1;                            % Re-trended prediction.
%                 varargout{5} = dly2;                          % Re-scaled, re-trended prediction.
            end

            function dlx = subsample_data(x)
                % function for subsampling and rearranging scaled data
                persistent set

                obj.debug_info("<strong>Subsampling Layer</strong>",[]);        % display active layer's name while in debug
                
                if isempty(set)                                                 % if persistent variable is empty initialize set as zeros
                    set = gpudl(zeros([obj.info.WindowSize obj.info.nSubsamples size(x,2)],obj.DataType),'');           % [ws nsubsamples batchsize]
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
            WeightCoeffs = ReweightDecoder(dlx,obj.weights.RDL.RD);                 % compute reweighing coefficients (outlier detection & removal)
            DetrendValues = DetrendDecoder(dlx,obj.weights.RDL.DD);                 % compute detrending values

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
                for i = 1:obj.info.RDL.RE.nBlocks                                                   % loop through layers
                    WeightCoeffs = decoder_block(WeightCoeffs,layer.tcn1d1,@tanh);                  % decoder block tanh activation

                    debug_message = append("Output size after ",num2str(i)," iterations ");         % debug message
                    obj.debug_info(debug_message,WeightCoeffs);                                     % display layer output size 
                end
            end

            function DetrendValues = DetrendDecoder(info,layer)
                % function for decoding encoder data into values for detrending

                obj.debug_info("<strong>Detrend Decoder</strong>",[]);                                  % display active layer's name while in debug


                DetrendValues = info;
                for i = 1:obj.info.RDL.DD.nBlocks                                                       % loop through layers
                    DetrendValues = decoder_block(DetrendValues,layer.tcn1d1,@tanh);                    % decoder block tanh activation

                    debug_message = append("Output size after ",num2str(i)," iterations ");             % debug message
                    obj.debug_info(debug_message,DetrendValues);                                        % display layer output size 
                end
            end

            function dly = encoder_block(dlx,layer,activation)
                % function for encoder block
                dly = obj.convlayer(obj.dropout(dlx),layer.cn1d1,'DataFormat','SCB');               % first 1D convolution
                dly = obj.convlayer(obj.dropout(dly),layer.cn1d2,'DataFormat','SCB');               % second 1D convolution

                dly = obj.maxpoollayer(obj.dropout(dly),[9],'Stride',[3],'DataFormat','SCB');       % max pooling layer

                dly = obj.batchnormlayer(obj.dropout(dly),layer.bn1,'DataFormat','SCB');            % batch normalization layer

                dly = activation(dly);                                                              % activation layer
            end

            function dly = decoder_block(dlx,layer,activation)
                % function for decoder block

                dly = transposedconvlayer(obj.dropout(dlx),layer.tcn1d1,'DataFormat','SCB');        % transposed 1D convolution
                dly = batchnormlayer(obj.dropout(dly),layer.bn1,'DataFormat','SCB');                % batch norm layer
                dly = activation(dly);                                                              % activation layer
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
            obj.debug_info("Output size after permute operation: ",dly);                        % display layer output size

            for i = 1:obj.info.SEL.nBlocks                                                      % loop through set encoding layer blocks
                dly = set_encodingblock(obj.dropout(dly),obj.weights.SEL.blocks{i},@relu);       % set encoding blocks with relu activation

                debug_message = append("Output size after ",num2str(i)," iterations ");         % debug message
                obj.debug_info(debug_message,DetrendValues);                                    % display layer output size 
            end

            function dly = set_encodingblock(dlx,layer,activation)
                % set sncoding blocks
                
                dly = obj.convlayer(dlx,layer.cn3d1,'DataFormat','SSSCB');                      % first 3D convolution
                dly = obj.convlayer(dly,layer.cn3d2,'DataFormat','SSSCB');                      % second 3D convolution

                dly = obj.maxpoollayer(dly,[9 9 9],'Stride',[1 1 1],'DataFormat','SSSCB');      % maximum pooling layer

                dly = obj.batchnormlayer(dly,layer.bn1,'DataFormat','SSSCB');                   % batch normalization layer
                dly = activation(dly);                                                          % activation layer
            end            
        end

        function [dly] = estimate_encodinglayer(obj,dlx)
            % layer for encoding previous estimate 

            obj.debug_info("<strong>Estimate Encoding Layer</strong>",[]);                           % display active layer's name while in debug

            dlx = permute(dlx,[1 3 2]);         % Estimate is in the format TB, permute such that dimensions are TCB (1 Channel)
            dly = obj.waveletlayer(dlx);        % Wavelet transform the batch of estimates

            obj.debug_info("Output size after wavelet layer: ",dly);                                % display layer output size

            for i = 1:obj.info.EEL.nBlocks                                                          % loop through blocks
                dly = estimate_encoderblock(obj.dropout(dly),obj.weights.EEL.blocks{i},@relu);      % estimate encoder blocks with relu activation
            end

            function dly = estimate_encoderblock(dlx,layer,activation)
                % function for estimate encoder block
                
                dly = obj.convlayer(dlx,layer.cn2d1,'DataFormat','SSCB');                   % first 2D convolution
                dly = obj.convlayer(dly,layer.cn2d2,'DataFormat','SSCB');                   % second 2D convolution
        
                dly = obj.maxpoollayer(dly,[9 9],'Stride',[1 1],'DataFormat','SSCB');       % max pooling layer

                dly = obj.batchnormlayer(dly,layer.bn1,'DataFormat','SSCB');                % batch normalization layer
                dly = activation(dly);                                                      % activation layer
            end
        end

        function [dly,obj] = predictionlayer(obj,dlx_set,dlx_estimate)
            % layer for predicting signal from encoded data

            obj.debug_info("<strong>Prediction Layer</strong>",[]);                           % display active layer's name while in debug

        end

        function [dly] = inverse_detrendlayer(obj,dlx,DetrendInfo)
            % layer which predicts and re-applies trend to prediction

            obj.debug_info("<strong>Inverse Detrend Layer</strong>",[]);                      % display active layer's name while in debug

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
                    layer.blocks{i} = init_encoder_block(layersizes(2*i-1:2*i,:));              % initialize layerblocks
                end
            end

            function layer = init_reweightdecoder(layersizes)
                % function for initializing reweight decoder

                layer.blocks = cell(info.RD.nBlocks,1);                                 % create cell array for blocks

                for i = 1:info.RD.nBlocks                                               % loop through blocks
                    layer.blocks{i} = init_RD_block(layersizes(i,:));                   % initialize layerblocks
                end
            end

            function layer = init_detrenddecoder(layersizes)
                % function for initializing detrend decoder

                layer.blocks = cell(info.DD.nBlocks,1);                                 % create cell array for blocks
            
                for i = 1:info.DD.nBlocks                                               % loop through blocks
                    layer.blocks{i} = init_DD_block(layersizes(i,:));                   % initialize layerblocks
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

            info.nblocks = 0.5*size(layersizes,1);                                                          % number of blocks is half the rows of layersizes
            assert(rem(info.nblocks,1)==0,"Number of rows in layersizes must be a factor of two\n");        % ensure the number of rows is even
            layer.blocks = cell(info.nblocks,1);                                                            % create cell array for blocks

            for i = 1:info.nblocks                                                                          % loop through blocks
                layer.blocks{i} = init_set_encodingblock(layersizes(2*i-1:2*i,:));                          % initialize blocks
            end

            function block = init_set_encodingblock(ls)            
                % function for initializing set encoder blocks

                block.cn3d1 = obj.init_convlayer(ls(1,1:3),la(1,4),ls(1,5),obj.DataType);                   % init first 3D conv layer
                block.cn3d2 = obj.init_convlayer(ls(2,1:3),ls(2,4),ls(2,5),obj.DataType);                   % init second 3D conv layer
                block.bn1   = obj.init_batchnormlayer(ls(2,5));                                             % init batchnorm layer
                           
            end
        end

        function [layer,info] = init_estimate_encodinglayer(obj,layersizes)
                        layer = [];
            info = [];

        end

        function [layer,info] = init_predictionlayer(obj,layersizes)
                        layer = [];
            info = [];

        end

        function [layer,info] = init_inversedetrendlayer(obj,layersizes)
                        layer = [];
            info = [];

        end

        function [layer,info] = init_inversescalinglayer(obj,layersizes)
                        layer = [];
            info = [];

        end        

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %       Other Functions


    end

end