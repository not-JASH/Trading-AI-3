classdef discriminator < DeepNetwork
    %{
        This network analyzes the spectral content of an input signal to
        determine if it is real o

    %}

    properties

    end

    methods 
        function obj = discriminator(LayerSizes,WindowSize)
                % constructor

                obj.info.WindowSize = WindowSize;               % store windowsize in network's info
            
                [~,obj.info.WTL] = obj.init_waveletlayer;       % initialize wavelet layer

        end

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %
        %   Layer Functions

        function dly = predict(obj,dlx)

        end

        function dly = waveletlayer(obj,dlx)

        end

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %
        %   Layer initialization functions

        function [layer,info] = init_waveletlayer(obj)
            % function for initializing wavelet layer

            wavelet = cwtfilterbank('SignalLength',obj.info.WindowSize,'Boundary','periodic');
            [info.psi_fvec,info.filter_idx] = cwtfilter2array(wavelet);

            layer = [];

        end

    end

end