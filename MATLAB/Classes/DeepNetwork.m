classdef DeepNetwork
    properties
        DataType = 'single';        %   Network Datatype
        IsTraining = true;          %   Boolean denoting if network is training
        Debug = false;              %   Boolean denoting network's debug state

        weights
        info
    end

    methods
         function debug_info(obj,message,item)
             % function for displaying debug info

             if obj.Debug                                               % check if network is in debug mode
                 if ~isempty(item)                                      % check if there's an item to evaluate
                    message = append(message,num2str(size(item)));      % append item's size to message
                 end
                 fprintf(append(message,"\n"));                         % display message
             end
         end

         function dly = dropout(obj,dly,varargin)
             % function for setting random array values to zero
            if ~obj.IsTraining;return;end       % if network is not training exit dropout function
            if ~isempty(varargin)               % if dropout rate is unspecified set it to 0.1
                rte = varargin{1};
            else
                rte = 0.1;
            end

            dly(randperm(numel(dly),floor(rte*numel(dly)))) = 0;   % set random array elements to zero 
         end
    end

    methods (Static)

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %   Convolution Layer (nD)
        %   
        %   For convolutions & grouped convolutions
        %   output_height = 1 + (input_height + padding_height_top + padding_height_bottom - filter_height)/(stride_height) 
        %   output_width = 1 + (output_width + padding_width_left + padding_width_right - filter_width)/(stride_width)

        function layer = init_convlayer(fs,nc,nf,datatype)
            % convolution layer initialization function 
            % fs    filter size (can be up to three-dimensional)
            % nc    number of channels
            % nf    number of filters

            layer.w = gpudl(init_gauss([fs nc nf],datatype),'');        % initialize layer weights
            layer.b = gpudl(zeros([nf 1],datatype),'');                    % initialize layer bias
        end

        function dly = convlayer(dlx,layer,varargin)                        
            % dl convolution layer
            dly = dlconv(dlx,layer.w,layer.b,varargin{:});              % convolution
        end

        function output_size = convoutputsize(fs,is,stride,varargin)
            % function for calculating output dimensions of convolution (& grouped conv) layers

            dimensionality  = ndims(fs);                % determine dimensionality of input
            output_size     = zeros(dimensionality,1);  % init output_size as zeros
            padding         = zeros(dimensionality,2);  % init padding as zeros
            
            if ~isempty(varargin)                       % if padding values were input
                for i = 1:nargin-3                      % loop through varargin
                    padding(i,:) = varargin{i};         % store padding values in matrix
                end
            end

            for i = 1:dimensionality
                output_size(i) = 1 + (is(i) + sum(padding(i,:)) - fs(i))/stride(i); % calculate output size of ith dimension
            end
        end

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %   Grouped Convolution Layer (nD)

        function layer = init_groupedconvlayer(fs,ncpg,nfpg,ng,datatype)
            % grouped convolution layer initialization function
            % fs        filter size (can be up to three-dimensional)
            % ncpg      number of convolutions per size
            % nfpg      number of filters per group
            % ng        number of groups

            layer.w = gpudl(init_gauss([fs ncpg nfpg ng],datatype),''); % initialize layer weights
            layer.b = gpudl(zeros([nfpg*ng 1],datatype),'');            % initialize layer bias
        end

        function dly = groupedconvlayer(dlx,layer,varargin)
            % grouped convolution layer
            dly = dlconv(dlx,layer.w,layer.b,varargin{:});      % convolution
        end

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %   Transposed Convolution Layer (nD)

        function layer = init_transposedconvlayer(fs,nc,nf,datatype)
            % initialize transposed convolution layer
            % fs    filters size (can be up to three-dimensional)
            % nc    number of channels
            % nf    number of filters

            layer.w = gpudl(init_gauss([fs nf nc],datatype),'');    % initialize layer weights
            layer.b = gpudl(zeros([nf 1],datatype),'');             % initialize layer bias
        end

        function dly = transposedconvlayer(dlx,layer,varargin)
            % transposed convolution layer
            dly = dltranspconv(dlx,layer.w,layer.b,varargin{:});    % transposed convolution
        end

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %   Grouped Transposed Convolution Layer (nD)

        function layer = init_groupedtransposedconvlayer(fs,ncpg,nfpg,ng,datatype)
            % initialize grouped transposed convolution layer
            % fs    filter size
            % ncpg  number of channels per group
            % nfpg  number of filters per group
            % ng    number of groups

            layer.w = gpudl(init_gauss([fs ncpg nfpg ng],datatype),''); % initialize layer weights
            layer.b = gpudl(zeros([nfpg*ng 1],datatype),'');            % initialize layer bias
        end

        function dly = groupedtransposedconvlayer(dlx,layer,varargin)
            % grouped transposed convolution layer
            dly = dltranspconv(dlx,layer.w,layerb.b,varargin{:});   % transposed convolution
        end

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %   Fully Connected Layer

        function layer = init_fullyconnectedlayer(dimensions,datatype)
            % fully connected layer initialization function
            layer.w = gpudl(init_gauss(dimensions,datatype),'');        % initialize layer weights
            layer.b = gpudl(zeros([dimensions(1) 1],datatype),'');      % initialize layer bias
        end

        function dly = fullyconnectedlayer(dlx,layer,varargin)
            % fully connected layer
            dly = fullyconnect(dlx,layer.w,layer.b,varargin{:});        % fully connected layer
        end

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %   Batch Normalization

        function layer = init_batchnormlayer(nChannels,datatype)
            % batch normalization layer initialization function
            layer.o  = gpudl(zeros([nChannels 1],datatype),'');     % initialize offset
            layer.sf = gpudl(ones([nChannels 1],datatype),'');      % initialize scale factors
        end

        function dly = batchnormlayer(dlx,weights,varargin)
            % batch normalization layer
            dly = batchnorm(dlx,weights.o,weights.sf,varargin{:});  % batch normalization
        end

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %   Maximum Pooling 
        
        function dly = maxpoollayer(dlx,poolsize,varargin)
            % max pooling layer
            dly = maxpool(dlx,poolsize,varargin{:});    % Max pool
        end


        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        

    end
end
    