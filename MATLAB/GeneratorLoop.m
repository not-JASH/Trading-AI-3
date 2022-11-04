%{
        Generator Training Loop -> Without MetaModel

        Jashua Luna
        November 2022
%}

getworkers;                                         % init worker pool 
nWorkers = parcluster('local').NumWorkers;          % store worker count

% training parameters
WindowSize = 80;
ExtrapolationLength = 15;
Overlap = 20;
nSubsamples = 30;

nSamples = 1e3;
BatchSize = 48;

assert(rem(nSamples,nWorkers-1)==0,"nworkers-1 must be a factor of nsamples\n");
wSamples = nSamples/(nWorkers-1);   % determine number of samples each worker will generate

spmd                                                                    % start spmd block
    if spmdIndex == 1                                                   % if worker id = 1
        gen = generator([],WindowSize,Overlap,nSubsamples,BatchSize);   % init generator
        spmdBroadcast(1,gen.weightless_copy);                           % broadcast weightless copy of generator to workers
        [xdata,ydata] = deal(cell(nSamples,1));                         % init xdata and ydata as empty cell arrays
        locs = [1:wSamples];                                            % locs for storing x and y data from workers
        for i = 1:nWorkers-1                                            % loop through workers
            xdata(locs) = spmdReceive(i,1);                             % receive and store xdata
            ydata(locs) = spmdReceive(i,2);                             % receive and store ydata
            locs = locs + wSamples;                                     % incriment locs 
        end        
    else                                                    % all other workers
        gen = spmdBroadcast(1);                             % receive empty generator from worker 1
        [xdata,ydata] = gen.get_trainingdata(wSamples);     % generate training data    
                                                            
        spmdSend(xdata,1,1);                                % send xdata to worker 1
        spmdSend(ydata,1,2);                                % send ydata to worker 1
    end
end