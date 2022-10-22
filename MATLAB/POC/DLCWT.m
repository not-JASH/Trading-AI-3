

ws = 480;

wavelet = cwtfilterbank('SignalLength',ws,'Boundary','periodic');
[psi_fvec,filter_idx] = cwtfilters2array(wavelet);

locs = [1:ws];

x = [0:0.01:4*3.14159];
trend = 0.01*x.^2;
y1 = sin(x) + 0.24*cos(x) + 0.1*sin(10*x) + trend;
y2 = cos(10*x) + 0.5*sin(x).*cos(x) - trend;

y = dlarray(cat(1,y1(locs),y2(locs)));
y = permute(y,[1 3 2]);     % CBT
y = cat(2,y,y,y);

y = permute(y,[3 1 2]); % TCB



dly = dlcwt(y,psi_fvec,filter_idx,'DataFormat','TCB');

ref1 = cwt(y1(locs),'FilterBank',wavelet);
ref2 = cwt(y2(locs),'FilterBank',wavelet);