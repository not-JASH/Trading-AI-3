%{
        Class for Discriminator

        Jashua Luna
        October 2022
%}

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
            
            obj.info.WindowSize = WindowSize;                                           % store windowsize in network's info
            
            [~,obj.info.WTL] = obj.init_waveletlayer;                                   % initialize wavelet layer
                
            [obj.weights.PL,obj.info.PL] = obj.init_predictionlayer(LayerSizes.PL);     % init prediction layer
        end

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %
        %   Layer Functions

        function dly = predict(obj,dlx)
            % function for forward pass on discriminator

            obj.debug_info("<strong>Prediction Layer</strong>",[]);                         % display active layer's name while in debug
            
            dly = obj.waveletlayer(permute(dlx,[1 3 2]));                                   % wavelet transform input data
            dly = cat(3,real(dly),imag(dly));
            
            for i = 1:obj.info.PL.nBlocks                                                   % loop through blocks
                dly = conv_maxpool(obj.dropout(dly),obj.weights.PL.blocks{i},@relu);        % convolution x2 + maxpool blocks

                debug_message = append("Output size after ",num2str(i)," iterations ");     % debug message
                obj.debug_info(debug_message,dly);                                          % display layer output size 
            end

            dly = squeeze(dly);     % remove singleton dimension

            function dly = conv_maxpool(dlx,layer,activation)   
                % function for convolution x2 + maxpool layer 

                dly = obj.convlayer(dlx,layer.cn2d1,'DataFormat','SSCB');           % first 2D conv layer
                dly = activation(dly);                                              % activation layer

                dly = obj.convlayer(dly,layer.cn2d2,'DataFormat','SSCB');           % second 2D conv layer
                dly = activation(dly);                                              % activation layer
                
                dly = obj.batchnormlayer(dly,layer.bn1,'DataFormat','SSCB');        % batch norm layer               
            end
        end

        function [dly] = waveletlayer(obj,dlx)
            % layer to wavelet transform input data

            obj.debug_info("<strong>Wavelet Layer</strong>",[]);                                    % display active layer's name while in debug

            dly = dlcwt(dlx,obj.info.WTL.psi_fvec,obj.info.WTL.filter_idx,'DataFormat','TCB');      % Wavelet transform subsampled sets
            dly = permute(dly,[1 4 2 3]);                                                           % dlcwt output is SCBT where S is filter dilation, permute to STCB

            obj.debug_info("Output size after Wavelet Layer: ",dly);                                % display layer output size
        end

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %
        %   Layer initialization functions

        function [layer,info] = init_waveletlayer(obj)
            % function for initializing wavelet layer

            wavelet = cwtfilterbank('SignalLength',obj.info.WindowSize,'Boundary','periodic');  % init wavelet filterbank
            [info.psi_fvec,info.filter_idx] = cwtfilters2array(wavelet);                        % convert filterbank into array and load into memory

            layer = []; % no layer weights
        end

        function [layer,info] = init_predictionlayer(obj,layersizes)
            % init prediction layer
            
            info.nBlocks = .5*size(layersizes,1);                                           % determine number of blocks    
            assert(rem(info.nBlocks,1)==0,"rows in layersizes must be a factor of two");

            layer.blocks = cell(info.nBlocks,1);                                            % init cell array for blocks
            for i = 1:info.nBlocks                                                          % loop through layer blocks
                layer.blocks{i} = init_predictionlayerblock(layersizes(2*i-1:2*i,:));       % init prediction layer block

                debug_message = append("Output size after ",num2str(i)," iterations ");     % debug message
                obj.debug_info(debug_message,info);                                         % display layer output size 
            end
            
            function block = init_predictionlayerblock(ls)
                % function for initializing prediction layer block

                block.cn2d1 = obj.init_convlayer(ls(1,1:2),ls(1,3),ls(1,4),obj.DataType);       % init first conv layer
                block.cn2d2 = obj.init_convlayer(ls(2,1:2),ls(2,3),ls(2,4),obj.DataType);       % init second conv layer

                block.bn1   = obj.init_batchnormlayer(ls(2,4),obj.DataType);                    % init batchnorm layer
            end
        end
    end
end 