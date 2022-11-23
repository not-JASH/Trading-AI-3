%{
        Class for Metamodel

        Jashua Luna
        October 2022
%}

classdef metamodel < DeepNetwork
    properties

    end

    methods
        function obj = metamodel()


        end

        function sizes = get_sizes(layertype,inputsize,outputsize,nblocks)
            % abstract function for determining a possible set of sizes for a given layertype, which takes inputs of one size and returns
            % outputs of another size. arguments are tentative, nblocks for now
        end

        function layersizes = generator(obj)



            function ls = reweight_detrend()

            end

            function ls = set_encoding()

            end

            function ls = estimate_encoding()

            end
        end

        function layersizes = discriminator(obj)

        end

    end
end