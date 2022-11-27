%{
        Proof of Concept 3

        Jashua Luna
        November 2022

        Spectral Content Changes With Time
%}

data = get_data;
data = data(:,5) - data(:,2);

ws = 100;
nsubsamples = 5;
overlap = 30;

total_windowsize = ws + (nsubsamples-1)*(ws-overlap);
loc = randi([total_windowsize length(data)],1);

sample = data([-total_windowsize+1:0]+loc);

subsamples = zeros(ws,nsubsamples);
slocs = [1:ws];
sc = cell(nsubsamples,1);

for i = 1:nsubsamples
    subsamples(:,i) = sample(slocs);
    slocs = slocs+ws-overlap;
        
    sc{i} = wvd(subsamples(:,i));
end


for i = 1:nsubsamples
    subplot(1,nsubsamples,i)
    surf(sc{i},'linestyle','none');
    view(2);
    xticklabels([]);yticklabels([]);
    ylim([1 ws]);
    xlabel(sprintf("t + %d",i-1));
end

% sgtitle("Wigner-Ville Distributions of Overlapping Consecutive Samples")
