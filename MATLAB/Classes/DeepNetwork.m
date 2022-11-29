%{
        Deep Network Superclass

        Jashua Luna
        October 2022
%}


classdef DeepNetwork
    properties
        DataType = 'single';        %   Network Datatype
        IsTraining = true;          %   Boolean denoting if network is training
        Debug = false;              %   Boolean denoting network's debug state

        weights
        info
    end

    methods
        function obj = DeepNetwork(learnrate)
            % superclass constructor

            obj.info.avg_g = [];            % initialize average gradient for adam
            obj.info.avg_sqg = [];          % initialize average squared gradient for adam
            obj.info.settings.decay   = 1 - 1e-2;    % set default decay rate for adam
            obj.info.settings.sqdecay = 1 - 1e-4;    % set default squared decay rate for adam
            obj.info.settings.lr      = learnrate;   % set learnrate

        end

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

        function output_size = convoutputsize(fs,is,nco,stride,varargin)
            % function for calculating output dimensions of convolution (& grouped conv) layers

            assert(length(stride)==length(fs),"stride must have the same dimensionality as fs");
            assert(length(is) == length(fs)+1,"input size must be the same as filter size plus one dimension for channels")

            dimensionality  = length(fs);               % determine dimensionality of input
            output_size     = zeros(1,dimensionality);  % init output_size as zeros
            padding         = zeros(dimensionality,2);  % init padding as zeros
            
            if ~isempty(varargin)                       % if padding values were input
                for i = 1:nargin-4                      % loop through varargin
                    padding(i,:) = varargin{i};         % store padding values in matrix
                end
            end

            for i = 1:dimensionality
                output_size(i) = 1 + floor((is(i) + sum(padding(i,:)) - fs(i))/stride(i)); % calculate output size of ith dimension
            end

            output_size = [output_size,nco];    % append number of output channels to output_size
            assert(all(output_size)>0,"Dimension error. Output size is invalid");
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

        function output_size = transposedconvoutputsize(fs,is,nco,stride,varargin)
            % function for calculating output dimensions of transposed convolution (and grouped transposed convolution) layers
            
            assert(length(stride)==length(fs),"stride must have the same dimensionality as fs");
            assert(length(is) == length(fs)+1,"input size must be the same as filter size plus one dimension for channels");

            dimensionality  = length(fs);               % determine dimensionality of input
            output_size     = zeros(1,dimensionality);  % init output_size as zeros
            cropping         = 'valid';                 % default to no cropping

            if ~isempty(varargin)                       % if cropping is specified, and set to same
                if strcmp(varargin{1},'same')           % change the value
                    cropping = 'same';
                end
            end

            for i = 1:dimensionality                                % loop through sample dimensions
                if strcmp(cropping,'valid')
                    output_size(i) = (is(i)-1)*stride(i) + fs(i);   % output size if cropping is valid (none)
                elseif strcmp(cropping,'same')
                    output_size(i) = is(i)*stride(i);               % output size if cropping is same -> ignore specific cropping sizes 
                end
            end               
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

        function outputsize = maxpooloutputsize(inputsize,poolsize,stride,varargin)
            % function for determining a maxpooling layer's outputsize

            assert(length(inputsize) == length(poolsize)+1,"inputsize must have one more dimension than poolsize (for channels)");
            
            nci = inputsize(end); % determine number of channels in data 

            outputsize = DeepNetwork.convoutputsize(poolsize,inputsize,nci,stride,varargin{:}); % the only difference between this output size and conv output size is nchannelsout = nchannels in 

        end


        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %   Other Functions

        function [data,count] = struct_tree2cell(data)
            count = 0;
            if isstruct(data)
                data = struct2cell(data);
                for i = 1:length(data)
                    [data{i},temp] = DeepNetwork.struct_tree2cell(data{i});
                    count = count+temp;
                end
            elseif iscell(data)
                for i = 1:length(data)
                    [data{i},temp] = DeepNetwork.struct_tree2cell(data{i});
                    count = count+temp;
                end
            elseif isdlarray(data)
                count = numel(data);
            end             
        end
    end
end
    