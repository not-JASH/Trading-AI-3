classdef generator
    properties
        weights
        info

        previous_estimate
    end

    methods
        function obj = generator(layersizes)
             % generator initialization function
             obj.info.dropout = true;

             [obj.weights.SL,obj.info.SL]     = obj.init_scalinglayer(layersizes.SL);
             [obj.weights.RDL,obj.info.SL]    = obj.init_reweightdetrendlayer(layersizes.RDL);
             [obj.weights.WTL,obj.info.WTL]   = obj.init_waveletlayer(layersizes.WTL);
             [obj.weights.SEL,obj.info.SEL]   = obj.init_set_encodinglayer(layersizes.SEL);
             [obj.weights.EEL,obj.info.EEL]   = obj.init_estimate_encodinglayer(layersizes.EEL);
             [obj.weights.PL,obj.info.PL]     = obj.init_predictionlayer(layersizes.PL);
        end
        
        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %       Network & Layer Functions

        function [dly,varargout] = predict(dlx,varargin)

        end

        function [dly,FineTuneCoeff,ScaleFactor] = scalinglayer(obj,x)
            % layer for scaling input data

            ScaleFactor = range(abs(x));                % scale input data by the range of its absolute values
            dlx = gpudl(x/ScaleFactor);                  
            
            FineTuneCoeff = FineTunesScaling(dlx);      % obtain fine tune coefficient
            dly = dlx*FineTuneCoeff;                    % scale dlx by fine tune coeff

            function dly = FineTuneScaling(dlx,layer)
                % function for determining fine tune coefficient

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

            end

            function WeightCoeffs = ReweightDecoder(dlx,info,layer)
                % function for decoding encoder data into reweighting
                % coefficients
                
            end

            function DetrendValues = DetrendDecoder(dlx,info,layer)
                % function for decoding encoder data into values for
                % detrending

            end
        end

        function [dly] = waveletlayer(obj,dlx)
            % layer to wavelet transform input data

        end

        function [dly] = set_encodinglayer(obj,dlx)

        end

        function [dly] = estimate_encodinglayer(obj,dlx)

        end

        function [dly,obj] = predictionlayer(obj,dlx)

        end

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %       Layer Initialization Functions

        function [layer] = init_scalinglayer(layersizes)
            
        end

        function layer = init_reweightdetrendlayer(layersizes)

        end

        function layer = init_waveletlayer(layersizes)

        end

        function layer = init_set_encodinglayer(layersizes)

        end

        function layer = init_estimate_encodinglayer(layersizes)

        end

        function layer = init_predictionlayer(layersizes)

        end
        
    end

end