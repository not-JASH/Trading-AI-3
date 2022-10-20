classdef DeepNetwork
    properties
        DataType = 'Single';
    end

    methods (Static)

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %   Convolution Layer (nD)

        function layer = init_convlayer(fs,nc,nf,datatype)
            layer.w = gpudl(init_gauss([fs nc nf],datatype),'');
            layer.b = gpudl(zeros([nf 1],datatype));
        end

        function dly = convlayer(dlx,layer,varargin)
            dly = dlconv(dlx,layer.w,layer.b,varargin{:});
        end

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %   Grouped Convolution Layer (nD)

        function layer = init_groupedconvlayer(fs,ncpg,nfpg,ng,datatype)
            layer.w = gpudl(init_gauss([fs ncpg nfpg ng],datatype),'');
            layer.b = gpudl(zeros([nfpg*ng 1],datatype),'');
        end

        function dly = groupedconvlayer(dlx,layer,varargin)
            dly = dlconv(dlx,layer.w,layer.b,varargin{:});
        end

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %   Transposed Convolution Layer (nD)

        function layer = init_transposedconvlayer(fs,nc,nf,datatype)
            layer.w = gpudl(init_gauss([fs nf nc],datatype),'');
            layer.b = gpudl(zeros([nf 1],datatype),'');
        end

        function dly = transposedconvlayer(dlx,layer,varargin)
            dly = dltranspconv(dlx,layer.w,layer.b,varargin{:});
        end

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %   Grouped Transposed Convolution Layer (nD)

        function layer = init_groupedtransposedconvlayer(fs,ncpg,nfpg,ng,datatype)
            layer.w = gpudl(init_gauss([fs ncpg nfpg ng],datatype),'');
            layer.b = gpudl(zeros([nfpg*ng 1],datatype),'');
        end

        function dly = groupedtransposedconvlayer(dlx,layer,varargin)
            dly = dltranspconv(dlx,layer.w,layerb.b,varargin{:});
        end

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %   Fully Connected Layer

        function layer = init_fullyconnectedlayer(dimensions,datatype)
            layer.w = gpudl(init_gauss(dimensions,datatype),'');
            layer.b = gpudl(zeros([dimensions(1) 1],datatype),'');
        end

        function dly = fullyconnectedlayer(dlx,layer,varargin)
            dly = fullyconnect(dlx,layer.w,layer.b,varargin{:});
        end

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %   Batch Normalization

        function layer = init_batchnormlayer(nChannels,datatype)
            layer.o  = gpudl(zeros([nChannels 1],datatype),'');
            layer.sf = gpudl(ones([nChannels 1],datatype),'');
        end

        function dly = batchnormlayer(dlx,weights,varargin)
            dly = batchnorm(dlx,weights.o,weights.sf,varargin{:});
        end
    

    end
end
    