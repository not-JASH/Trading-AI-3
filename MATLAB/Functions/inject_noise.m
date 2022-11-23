function data = inject_noise(data)
    % adds noise to an array
    data = data + normrnd(0,0.035,size(data));
end