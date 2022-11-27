%{
        Generator Training Loop -> Without MetaModel

        Jashua Luna
        November 2022
%}

getworkers(6);                                      % init worker pool 
nWorkers = gcp('nocreate').NumWorkers;              % store worker count

% training parameters
WindowSize = 80;
ExtrapolationLength = 15;
Overlap = 20;
nSubsamples = 64;
nEvalSamples = 4;

nSamples = 1e3;
BatchSize = 30;

[lrg,lrd] = deal(3e-3);     % set generator and discriminator learn rates

assert(rem(nSamples,nWorkers-1)==0,"nworkers-1 must be a factor of nsamples\n");
wSamples = nSamples/(nWorkers-1);   % determine number of samples each worker will generate


%% Initialize networks and training data on workers

spmd                                                                    % start spmd block  
    if spmdIndex == 1                                                   % if worker id = 1
        gen = generator([],WindowSize,ExtrapolationLength,Overlap,nSubsamples,BatchSize,lrg);   % init generator
        disc = discriminator([],WindowSize,lrd);                        % init discriminator
        spmdBroadcast(1,gen.weightless_copy);                           % broadcast weightless copy of generator to workers
        [xdata,ydata] = deal(cell(nSamples,1));                         % init xdata and ydata as empty cell arrays
        locs = [1:wSamples];                                            % locs for storing x and y data from workers
        for i = 2:nWorkers                                              % loop through workers
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

%% Training Loop 

iter = 1;               % start counting iterations at zero
total_time = 0;         % set timers to zero
iteration_time = 0;     %

while true                                                                                  % open training loop
    tic                                                                                     % start counting iteration time
    spmd                                                                                    % open spmd block                    
        if spmdIndex == 1                                                                   % on worker 1
            batchlocs = [1:BatchSize];                                                      % initialize batch locs   
            
            while ~isempty(batchlocs)                                                       % loop through samples in batches
                [xbatch,ybatch] = gen.get_batch(xdata(batchlocs),ydata(batchlocs));         % get xbatch and ybatch
                [grad_gen,grad_disc] = dlfeval(@gen.model_gradients,xbatch,ybatch,disc,disc.weights,gen.weights);    % compute gradients for generator and discriminator
                [gen,disc] = gen.update_weights(grad_gen,grad_disc,disc,iter);              % update model weights 

                batchlocs = batchlocs + BatchSize;      % increment batchlocs
                batchlocs(batchlocs>nSamples) = [];     % clear locs that exceed nSamples
            end

            % evaluate performance after iteration
            eval_locs = randi(nSamples,[nEvalSamples 1]);                                   % evaluate performance on n samples
            [eval_x,eval_y] = gen.get_batch(xdata(eval_locs),ydata(eval_locs));             % generate batch for evaluation
    
            eval_dly = gen.predict(eval_x,inject_noise(eval_x(end-WindowSize+1,:,:)));      % predict with evaluation sample
            eval_y = eval_y.y;                                                              % discard extra data
               
            spmdBarrier;                            % pause execution until other workers have generated samples
            locs = [1:wSamples];                    % initialize locs up to wSamples
            for i = 2:nWorkers                      % loop through workers
                xdata(locs) = spmdReceive(i,1);     % receive and store xdata
                ydata(locs) = spmdReceive(i,2);     % receive and store ydata
                locs = locs + wSamples;             % increment locs
            end
        else                                                    % on all workers except worker 1
            [xdata,ydata] = gen.get_trainingdata(wSamples);     % generate training samples
            spmdBarrier;                                        % pause execution until all workers have generated samples and training iteration is complete

            spmdSend(xdata,1,1);                    % send xdata to worker 1
            spmdSend(ydata,1,2);                    % send ydata to worker 1
        end
    end

    eval_prediction = gatext(eval_dly{1});  % extract data from composite arrays
    eval_reference = gatext(eval_y{1});     % also remove arrays from gpu and make untraced

    for i = 1:nEvalSamples                      % loop through evaluation samples
        subplot(nEvalSamples,1,i)               % plot each sample on a separate subplot                          
        plot(eval_reference(:,i));              % plot reference sample
        hold on 
        plot(eval_prediction(:,i));             % plot predicted output
        hold off
    end
    f = getframe;
    
    iteration_time = toc;                           % how long did this iteration take
    total_time = iteration_time + total_time;       % add iteration time to total time
    [d,h,m,s] = gettimestats(total_time);           % calculate time in days hours minutes seconds
    fprintf("Iteration %d Complete, Time Elapsed: %.2f s\n",iter,iteration_time);   % display iteration info
    fprintf("Total Time Elapsed: %s:%s:%s:%s\n\n",d,h,m,s);                         % display training time info
    iter = iter+1;  % increment iteration
end
