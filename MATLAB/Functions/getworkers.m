function getworkers(varargin)
    % function for starting a worker pool
    
    if nargin>0         %if nworkers is not specified use max available
        nworkers = varargin{1};
    else
        nworkers = parcluster('local').NumWorkers;
    end

    % initialize pool with nworkers
    if isempty(gcp('nocreate')) 
        parpool(nworkers);
    end
end