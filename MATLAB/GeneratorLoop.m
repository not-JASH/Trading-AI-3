%{
        Generator Training Loop -> Without MetaModel
        
        Trains on a limited set of data

        Jashua Luna
        November 2022
%}

nworkers = 8;
getworkers(nworkers);

WindowSize = 80;
ExtrapolationLength = 30;
Overlap = 20;
nSubsamples = 64;
nEvalSamples = 10;

nSamples = .5e4;
BatchSize = 30;

[lrg,lrd] = deal(9e-3);     % set generator and discriminator learn rates

%% Initialize networks and generate training data

gen = generator([],WindowSize   ,ExtrapolationLength,Overlap,nSubsamples,BatchSize,lrg);   % init generator
disc = discriminator([],WindowSize,lrd);                                                % init discriminator

assert(rem(nSamples,1e3)==0,"1000 must be a factor of nsamples");   
[xdata,ydata] = deal(cell(nSamples/1e3,1));         % make cell arrays for xdata and ydata

datagen = @gen.get_trainingdata;    % store data generator as function to avoid overhead in parfor loop 

parfor i = 1:nSamples/1e3
    [xdata{i},ydata{i}] = datagen(1e3);     % generate samples
end

xdata = cat(1,xdata{:});    % reshape xdata
ydata = cat(1,ydata{:});    % reshape ydata

%% Training loop

epoch = 1;
total_time = 0;
iteration_time = 0;

genbatch = @gen.get_batch;

fprintf("Training Network\n");

while true 
    tic

    shuffle_locs = randperm(nSamples);      % shuffle samples each epoch
    batchlocs = [1:BatchSize];              % initialize batch locations
    
    while ~isempty(batchlocs)                                                                               % loop through samples in batches
        [xbatch,ybatch] = genbatch(xdata(shuffle_locs(batchlocs)),ydata(shuffle_locs(batchlocs)));          % get xbatch and ybatch
        [grad_gen,grad_disc] = dlfeval(@gen.model_gradients,xbatch,ybatch,disc,disc.weights,gen.weights);   % compute gradients for generator and discriminator
        [gen,disc] = gen.update_weights(grad_gen,grad_disc,disc,epoch);                                      % update model weights

        if any(rem(batchlocs,1e3)==0)
            % evaluate performance every 1000 samples
            eval_locs = randi(nSamples,[nEvalSamples 1]);                       % evaluate performance on n samples
            [eval_x,eval_y] = genbatch(xdata(eval_locs),ydata(eval_locs));      % generate batch for evaluation
        
            eval_prediction = gatext(gen.predict(eval_x,inject_noise(eval_x(end-WindowSize+1,:,:))));      % predict with evaluation sample
            eval_reference  = gatext(eval_y.y);                                                            % discard extra data
        
            for i = 1:nEvalSamples                      % loop through evaluation samples
                subplot(nEvalSamples,2,2*i-1)           % plot each sample on a separate subplot                          
                plot(eval_reference(:,i));              % plot reference sample
                hold on 
                plot(eval_prediction(:,i)/range(abs(eval_prediction(:,i))));        % plot scaled predicted output
                hold off
                xline(WindowSize-ExtrapolationLength);  % draw a line at the start of the extrapolated section
        
                subplot(nEvalSamples,2,2*i)             
                plot(cumsum(eval_reference(:,i)));      % plot the cumulative sum of reference samples
                hold on 
                plot(cumsum(eval_prediction(:,i)/range(abs(eval_prediction(:,i)))));    % plot cumulative sum of scaled predicted output
                hold off
                xline(WindowSize-ExtrapolationLength);  % draw a line at the start of the extrapolated section
            end
            f = getframe;
        end
       
        batchlocs = batchlocs + BatchSize;      % increment batchlocs
        batchlocs(batchlocs>nSamples) = [];     % clear locs that exceed nSamples    
    end


    
    iteration_time = toc;                           % how long did this iteration take
    total_time = iteration_time + total_time;       % add iteration time to total time
    [d,h,m,s] = gettimestats(total_time);           % calculate time in days hours minutes seconds
    fprintf("Iteration %d Complete, Time Elapsed: %.2f s\n",epoch,iteration_time);   % display iteration info
    fprintf("Total Time Elapsed: %s:%s:%s:%s\n\n",d,h,m,s);                         % display training time info
    
    
    if rem(epoch,5) == 0                                                                         % every n iterations
        checkpointsave_generator = gen;                                                          % extract generator from parallel pool  
        checkpointsave_discriminator = disc;                                                     % extract discriminator from parallel pool

        save("checkpointsave.mat","checkpointsave_discriminator","checkpointsave_generator","epoch","total_time");       % save generator, discriminator, iter, and total_time
    end

    epoch = epoch+1;  % increment iteration    

end


