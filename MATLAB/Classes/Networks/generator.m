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

        function [dly,info] = scalinglayer(obj,dlx)

        end

        function [dly,info] = reweightdetrendlayer(obj,dlx)

        end

        function [dly] = waveletlayer(obj,dlx)

        end

        function [dly] = set_encodinglayer(obj,dlx)

        end

        function [dly] = estimate_encodinglayer(obj,dlx)

        end

        function [dly,obj] = predictionlayer(obj,dlx)

        end

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %       Layer Initialization Functions

        function layer = init_scalinglayer(layersizes)

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