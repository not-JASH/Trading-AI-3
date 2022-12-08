%{
        Generator Training Loop Debug -> Without MetaModel

        Jashua Luna
        November 2022

        script for evaluating generator object's data training functions. 
%}

% training parameters
WindowSize = 80;
ExtrapolationLength = 15;
Overlap = 20;
nSubsamples = 30;

nSamples = 1e2;
BatchSize = 48;

[lrg,lrd] = deal(3e-3);     % set generator and discriminator learn rates

gen = generator([],WindowSize,ExtrapolationLength,Overlap,nSubsamples,BatchSize,lrg);       % create generator object
blank_gen = gen.weightless_copy;
gen.Debug = true;

disc = discriminator([],WindowSize,lrd);                                                    % create discriminator object
disc.Debug = true;

[xdata,ydata] = blank_gen.get_trainingdata(nSamples);                                   % generate samples for training


batchlocs = 1:BatchSize;                                                                              % init batch locs
[xbatch,ybatch] = gen.get_batch(xdata(batchlocs),ydata(batchlocs));                                     % get batches from data
[grad_gen,grad_disc] = dlfeval(@gen.model_gradients,xbatch,ybatch,disc,disc.weights,gen.weights);       % get gradients
[gen,disc] = gen.update_weights(grad_gen,grad_disc,disc,1);                                             % update weights

