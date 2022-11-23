function [grad_gen,grad_disc] = modelgradients(gen_loss,disc_loss,gen,disc)
    
    grad_gen = dlgradient(gen_loss,gen,'RetainData',true);
    grad_disc = dlgradient(disc_loss,disc);
end