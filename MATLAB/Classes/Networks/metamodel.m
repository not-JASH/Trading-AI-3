%{
        Class for Metamodel

        Jashua Luna
        October 2022
%}

classdef metamodel < temp_DeepNetwork
    properties

    end

    methods
        function obj = metamodel(learnrate,WindowSize,nSubsamples)
            % class constructor for metamodel

            obj = obj@temp_DeepNetwork(learnrate);  % call superclass constructor

            obj.info.WindowSize = WindowSize;       % store windowsize in object
            obj.info.nSubsamples = nSubsamples;     % store nsubsamples in object

        end

        function outputsize = get_blockoutputsize(obj,layerinfo,inputsize)
            % function for determining the output size of a block containing several layers

            nlayers = length(layerinfo);
            outputsize = inputsize;
         
            for i = 1:nlayers
                switch layerinfo{i}.type
                    case 'conv'         % if layer is convolution
                        outputsize = obj.convoutputsize(layerinfo{i}.fs,outputsize,layerinfo{i}.nco,layerinfo{i}.stride);
                    case 'conv_same'
                        outputsize(end) = layerinfo{i}.nco; % the only dimension that changes is channel dimension
                    case 'tconv'        % if layer is transposed convolution
                        outputsize = obj.transposedconvoutputsize(layerinfo{i}.fs,outputsize,layerinfo{i}.nco,layerinfo{i}.stride);
                    case 'maxpool'      % if layer is maxpooling
                        outputsize = obj.maxpooloutputsize(outputsize,layerinfo{i}.poolsize,layerinfo{i}.stride);
                    case 'squeeze'      % if layer is squeeze 
                        outputsize(outputsize == 1) = [];   
                    case 'permute'      % if layer is permute
                        outputsize = outputsize(layerinfo{i}.order);
                    case 'layerblock'   % if layer is a series of layerblocks
                        for j = 1:layerinfo.nblocks
                            outputsize = obj.get_blockoutputsize(layerinfo{i}.layerinfo{j},outputsize);
                        end
                    otherwise           % all other layers (that this network might use) do not affect output size
%                         outputsize = outputsize;
                end
            end
        end

        function outputsize = check_generator(obj,metaparams)
            % function for verifying generator metaparameters and determining outputsize

            inputsize = [obj.info.WindowSize obj.info.nSubsamples]; % skip scaling and subsampling layer. subsampling layer output has dimensions [windowsize nsumsamples]
            
            [outputsize,dtisize]    = obj.verifylayer_reweightdetrend(inputsize,metaparams.RDL);                        % verify reweight detrend layer
            outputsize              = obj.verifylayer_wavelet(outputsize);                                              % verify wavelet layer
            outputsize              = obj.verifylayer_setencoding(outputsize,metaparams.SEL);                           % set encoding layer
            estimateoutputsize      = obj.verifylayer_estimateencoding([obj.windowsize 1],metaparams.EEL);              % estimate encoding layer
            outputsize              = obj.verifylayer_predictionlayer(outputsize,estimateoutputsize,metaparams.PL);     % prediction layer
            outputsize              = obj.verifylayer_inversedetrendlayer(outputsize,dtisize,metaparams.IDL);           % inverse detrend layer
            
        end
        
        function [outputsize,detrendinfosize] = verifylayer_reweightdetrend(obj,inputsize,metaparams)
            % function for verifying reweightdetrend metaparameters & determining output size
            
            InputEncoder        = obj.init_layerinfo({'conv','conv','maxpool'},metaparams.IE);  % generate layerinfo array for input encoder
            ReweightDecoder     = obj.init_layerinfo({'conv'},metaparams.RD);                   % generate layerinfo array for reweight decoder
%             DetrendDecoder      = obj.init_layerinfo({'conv'},metaparams.DD);                   % generate layerinfo array for detrend decoder (same as reweight decoder)
            
            outputsize = cell(2,1);     % this layer has two outputs

            detrendinfosize    = obj.get_blockoutputsize(InputEncoder,inputsize);        % get output size for input encoder
            outputsize         = obj.get_blockoutputsize(ReweightDecoder,outputsize{1}); % get output size for reweight decoder / detrend decoder

            assert(all(size(outputsize)==size(inputsize)),"decoder output must be the same size as input");
            
        end

        function outputsize = verifylayer_wavelet(obj,inputsize)
            % function for verifying waveletlayer metaparameters & determining output size

            wavelet = cwtfilterbank('SignalLength',obj.info.WindowSize,'Boundary','periodic');      % cwt filterbank for windowsize
            [psi,idx] = cwtfilters2array(wavelet);                                                  % convert to arrays

            outputsize = size(dlcwt(dlarray(rand(inputsize),''),psi,idx,'DataFormat','TCB'));       % perform cwt on sample with same size as input to get output size
            outputsize = outputsize([1 4 2 3]);     % rearrange output size [TCCB]
        end

        function outputsize = verifylayer_setencoding(obj,inputsize,metaparams)
            % function for verifying set encoding layer metaparameters & determining output size

            assert(length(inputsize)==3,'set encoding layer input must be three dimensional');  % batch dimension excluded 

            encoder = obj.init_layerinfo({'conv','conv','maxpool'},metaparams); % generate layerinfo array for encoder
            
            outputsize = [inputsize 2];                                 % split into real and complex components and concatenate along fourth dimension
            outputsize = obj.get_blockoutputsize(encoder,outputsize);   % get output size for encoder
            outputsize(outputsize==1) = [];                             % squeeze 

        end

        function outputsize = verifylayer_estimateencoding(obj,inputsize,metaparams)
            % function for verifying estimate encoding layer metaparameters & determining output size

            encoder = obj.init_layerinfo({'conv','conv','maxpool'},metaparams);

            outputsize = obj.verifylayer_wavelet(inputsize);            % encoding layer wavelet transform
            outputsize = obj.get_blockoutputsize(encoder,outputsize);   % encoder output size

        end

        function outputsize = verifylayer_predictionlayer(obj,setinputsize,estimateinputsize,metaparams)
            % function for verifying prediction layer metaparameters & determining output size
            
            setdecoder = obj.init_layerinfo({'tconv'},metaparams.SD);        % layerinfo array for set decoder
            predictiondecoder = obj.init_layerinfo({'conv_same'},metaparams.PD);  % layerinfo array for prediction decoder
            
            outputsize = obj.get_blockoutputsize(setdecoder,setinputsize);  % get outputsize for set decoder

            assert(all(size(outputsize) == size(estimateinputsize)),"setdecoder output must be the same size as estimate encoder input");   
            assert(rem(prod(outputsize),obj.info.WindowSize)==0,"WindowSize must be a factor of numel(decoder output) for reshape operation to function")

            outputsize  = [obj.info.WindowSize prod(outputsize)/obj.info.WindowSize];    % reshape layer
            outputsize  = obj.get_blockoutputsize(predictiondecoder,outputsize);         %  get outputsize for prediction decoder
            
            outputsize(outputsize==1) = []; % squeeze layer                

        end

        function outputsize = verifylayer_inversedetrendlayer(obj,inputsize,detrendinfosize,metaparams)
            % function for verifying inverse detrend layer metaparameters & determining output size

            detrend_decoder = obj.init_layerinfo({'tconv'},metaparams.DD);  % init layerinfo array for detrend decoder

            outputsize = obj.get_blockoutputsize(detrend_decoder,detrendinfosize);  % get outputsize for detrend decoder
            
            assert(all(size(inputsize) == size(outputsize)),"detrended prediction must be the same size as detrend decoder output");

        end

        function layerinfo = init_layerinfo(obj,layertypes,metaparams)
            % function for converting metaparams into layerinfo arrays based on specified layertypes

            layerinfo = cell(metaparams.nblocks,1);     % initialize layerinfo as a cell array which can accomodate each layer
            
            for b = 1:metaparams.nblocks
                layerinfo{b} = cell(size(layertypes));  % initialize a cell array for storing each block's layer's info
                
                for i = 1:length(layertypes)
                    switch layertypes{i}
                        case 'conv'
                            % convolution parameter matricies are structured [fs nci nco]
                            layerparams = metaparams.(append('conv',num2str(count_occurances('conv'))));     % read layerinfo from metaparameters, adjust field name for nth occurances of a layertype
                            params = layerparams.params;                                            % load size parameters from layerparameters

                            layerinfo{b}{i}.fs = params(1:layerparams.nd);      % load filtersize
                            layerinfo{b}{i}.nco = params(layerparams.nd+2);     % load nchannels out

                            layerinfo{b}{i}.stride = layerparams.stride;        % load stride

                        case 'conv_same'
                            layerparams = metaparams.(append('conv_same',num2str(count_occurances('conv_same'))));     % read layerinfo from metaparameters, adjust field name for nth occurances of a layertype
                            params = layerparams.params;                        % load size parameters from layerparameters
                            % this layer also has nchannels in and filtersize but those parameters are not required for getting output size
                            layerinfo{b}{i}.nco = params(layerparams.nd+2);     % load nchannels out

                        case 'tconv'
                            layerparams = metaparams.(append('tconv',num2str(count_occurances('tconv'))));     % read layerinfo from metaparameters, adjust field name for nth occurances of a layertype
                            params = layerparams.params;                        % load size parameters from layerparameters

                            layerinfo{b}{i}.fs = params(1:layerparams.nd);      % load filtersize
                            layerinfo{b}{i}.nco = params(layerparams.nd+2);     % load nchannels out

                            layerinfo{b}{i}.stride = layerparams.stride;        % load stride

                        case 'maxpool'
                            layerparams = metaparams.(append('maxpool',num2str(count_occurances('maxpool'))));     % read layerinfo from metaparameters, adjust field name for nth occurances of a layertype
                            
                            layerinfo{b}{i}.poolsize = layerparams.poolsize;    % pool size
                            layerinfo{b}{i}.stride = layerparams.stride;        % stride
                    end
                end
            end          

            function count = count_occurances(string)
                % function for counting the occurances of a string within layertypes, up to a certain point

                count = 0;

                for j = 1:i
                    if layertypes{j} == string;count=count+1;end
                end
            end
        end
    end
end


































