%{
        Proof of Concept 2

        Jashua Luna
        November 2022

        Varying structures apparent during different market conditions.
%}

data = get_data;            % load data from file
data = data(:,5)-data(:,2); % close - open = change in price / candle


nsamples = 1e2;         % number of samples
windowsize = 100;       % windowsize

samples = cell(nsamples,1);     % init cell array for samples
energy = zeros(nsamples,1);     % array for energy values of samples

x = [-windowsize+1:0];          % template locs for sample window 
locs = randi([windowsize size(data,1)],[nsamples 1]);   % sample locs

for i = 1:nsamples
    samples{i} = data(locs(i)+x);           % store samples in cell array
    energy(i) = sum(abs(fft(samples{i})));  % calculate then store energy readings
end
    
[~,he] = max(energy);   % location of high energy sample
[~,le] = min(energy);   % location of low energy sample


