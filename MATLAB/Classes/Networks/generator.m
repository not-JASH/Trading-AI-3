classdef generator < DeepNetwork
    properties
        weights
        info

        previous_estimate
    end

    methods
        function obj = generator(layersizes)
             % generator initialization function
             obj.info.dropout   = true;
             obj.info.training  = true;

             [obj.weights.SL,obj.info.SL]       = obj.init_scalinglayer(layersizes.SL);                 % Initialize scaling layer.
             [obj.weights.RDL,obj.info.RDL]     = obj.init_reweightdetrendlayer(layersizes.RDL);        % Initialize reweight-detrend layer.
             [obj.weights.WTL,obj.info.WTL]     = obj.init_waveletlayer(layersizes.WTL);                % Initialize wavelet layer.
             [obj.weights.SEL,obj.info.SEL]     = obj.init_set_encodinglayer(layersizes.SEL);           % Initialize set-encoder.
             [obj.weights.EEL,obj.info.EEL]     = obj.init_estimate_encodinglayer(layersizes.EEL);      % Initialize estimate-encoder.
             [obj.weights.PL,obj.info.PL]       = obj.init_predictionlayer(layersizes.PL);              % Initialize prediction layer.
             [obj.weights.IDL,obj.info.IDL]     = obj.init_inversedetrendlayer(layersizes.IDL);         % Initialize inverse-detrend layer.
             [obj.weights.ISL,obj.info.ISL]     = obj.init_inversescalinglayer(layersizes.ISL);         % Initialize inverse-scaling layer.

        end
        
        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %       Network & Layer Functions

        function [dly,varargout] = predict(obj,x,varargin)
            % feed forward function

            if obj.info.training                    % If network is training, xhat is supplied as training data.
                xhat = varargin{1};
            else                                    % Otherwise, xhat is stored in the network. 
                xhat = obj.previous_estimate;
            end
            
            [x,FineTuneCoeff,ScaleFactor]       = obj.scalinglayer(x);                                              % Scale input data 
            dlx                                 = subsample_data(x,obj.info.windowsize,obj.info.overlap);           % Subsample data
            [dlx,ReweightDetrendInfo]           = obj.reweightdetrendlayer(dlx);                                    % Reweight and detrend subsampled data.
            dly_set                             = obj.waveletLayer(dlx);                                            % Wavelet transform dataset.
            dly_set                             = obj.set_encodinglayer(dly_set);                                   % Encode input dataset. 
            dly_estimate                        = obj.estimate_encodinglayer(xhat);                                 % Encode previous estimate.
            dly_estimate                        = obj.predictionlayer(dly_set,dly_estimate);                        % Predict future system states from encoded data.
            dly1                                = obj.inverse_detrendlayer(dly_estimate,ReweightDetrendInfo);       % Re-apply trend. 
            dly2                                = obj.inverse_scalinglayer(dly1,FineTuneCoeff,ScaleFactor);         % Re-scale predicted values. 
            
            dly = dly2;

            if ~obj.info.training                               % If network is not training store prediction and return self
                obj.previous_estimate = dly_estimate;           % It would be more efficient to have two separate functions for training and use
                varargout{1} = obj;
            else                                                % Otherwise, return all values relevant to the loss function.
                varargout{1} = x;                               % Scaled data.
                varargout{2} = dlx2;                            % Reweighted and Detrended Data.
                varargout{3} = dly_estimate;                    % Un-scaled prediction.
                varargout{4} = dly1;                            % Re-trended prediction.
                varargout{5} = dly2;                            % Re-scaled, re-trended prediction.
            end

            function dlx = subsample_data(x,windowsize,overlap)
                % function for subsampling and rearranging scaled data

            end
        end

        function [dly,FineTuneCoeff,ScaleFactor] = scalinglayer(obj,x)
            % layer for scaling and subsampling input data

            ScaleFactor = range(abs(x));                % scale input data by the range of its absolute values
            dlx = gpudl(x/ScaleFactor);                  
            
            FineTuneCoeff = FineTunesScaling(dlx);      % obtain fine tune coefficient
            dly = dlx*FineTuneCoeff;                    % scale dlx by fine tune coeff

            function dly = FineTuneScaling(dlx,layer)
                % function for determining fine tune coefficient
                dly = fc_layer(obj.dropout(dlx),layer.block{1},'CB',@relu);         % fully connected layer on scaled values, relu activation
                for i = 2:obj.info.SL.FTS.nBlocks                                   % loop through layers
                    dly = fc_layer(obj.dropout(dlx),layerb.block{i},'CB',@relu);    % fully connected layer, relu activation
                end
            end
        end

        function [dly,info] = reweightdetrendlayer(obj,dlx)
            % layer for reweighting and detrending input data

            info = InputEncoder(dlx,obj.RDL.InputEncoder);                          % encode input data
            WeightCoeffs = ReweightDecoder(dlx,info,obj.RDL.ReweightDecoder);       % compute reweighing coefficients (outlier detection & removal)
            DetrendValues = DetrendDecoder(dlx,info,obj.RDL.DetrendDecoder);        % compute detrending values

            dly = dlx.*WeightCoeffs + DetrendValues;                                % reweight and detrend data

            function info = InputEncoder(dlx,layer)
                % layer for encoding input data into relevant information

                info = convlayer(obj.dropout(dlx),layer.block{1},'CB');             % 1D convolution on input data
                info = relu(info);                                                  % activation

                for i = 2:obj.info.RDL.IE.nBlocks                                   % loop through layers
                    info = convlayer(obj.dropout(info),layer.block{i},'SCB');       % 1D convolution 
                    info = relu(info);                                              % activation
                end
            end

            function WeightCoeffs = ReweightDecoder(info,layer)
                % function for decoding encoder data into reweighting
                % coefficients

                info = fc_layer(obj.dropout(info),layer.block{1},'SCB',@relu);      % fully connected layer on encoder output, relu activation
                
                for i = 2:obj.info.RDL.RE.nBlocks                                   % loop through layers
                    info = fc_layer(obj.dropout(info),layer.block{i},'CB',@relu);   % fully connected layer (batchnorm?), relu activation
                end
                WeightCoeffs = info;
            end

            function DetrendValues = DetrendDecoder(info,layer)
                % function for decoding encoder data into values for
                % detrending
                
                info = fc_layer(obj.dropout(info),layer.block{1},'SCB',@relu);      % fully connected layer on encoder output, relu activation

                for i = 2:obj.info.RDL.DD.nBlocks                                   % loop through layers
                    info = fc_layer(obj.dropout(info),layer.block{i},'CB',@relu);   % fully connected layer (batchnorm?), relu activation
                end
                DetrendValues = info;
            end

            function dly = fc_layer(dlx,layer,dataformat,activation)
                % function specific wrapper for fullyconnected layer
                dly = fullyconnectedlayer(dlx,layer.fc1,'DataFormat',dataformat);   % fully connected layer
                dly = activation(dly);                                              % activation function
            end
        end

        function [dly] = waveletlayer(obj,dlx)
            % layer to wavelet transform input data

        end

        function [dly] = set_encodinglayer(obj,dlx)

        end

        function [dly] = estimate_encodinglayer(obj,dlx)

        end

        function [dly,obj] = predictionlayer(obj,dlx_set,dlx_estimate)

        end

        function [dly] = inverse_detrendlayer(obj,dlx,DetrendInfo)

        end

        function [dly] = inverse_scalinglayer(obj,dlx,FineTuneCoeff,ScaleFactor)

        end

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %       Layer Initialization Functions

        function [layer] = init_scalinglayer(obj,layersizes)
            
        end

        function layer = init_reweightdetrendlayer(obj,layersizes)

        end

        function layer = init_waveletlayer(obj,layersizes)

        end

        function layer = init_set_encodinglayer(obj,layersizes)

        end

        function layer = init_estimate_encodinglayer(obj,layersizes)

        end

        function layer = init_predictionlayer(obj,layersizes)

        end

        function layer = init_inversedetrendlayer(obj,layersizes)

        end

        function layer = init_inversescalinglayer(obj,layersizes)

        end        

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %       Other Functions

        function data = dropout(obj,data)
            
        end
    end

end