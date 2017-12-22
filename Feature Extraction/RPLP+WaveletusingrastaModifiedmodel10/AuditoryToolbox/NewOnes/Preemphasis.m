function [y]=Preemphasis(x)
    B=[1 -.97]; % can be .95 to .99
    y= filter(B,1,x);
end
