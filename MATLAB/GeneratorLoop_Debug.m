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

nSamples = 1e3;
BatchSize = 48;

gen = generator([],WindowSize,ExtrapolationLength,Overlap,nSubsamples,BatchSize);       % create generator object
[xdata,ydata] = gen.get_trainingdata(nSamples);                                         % generate samples for training


