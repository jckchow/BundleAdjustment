function q = quaternionExp(w)

wx = w(1);
wy = w(2);
wz = w(3);

w_norm = sqrt(wx^2 + wy^2 + wz^2);

q = [cos(0.5*w_norm); wx/w_norm * sin(0.5*w_norm); wy/w_norm * sin(0.5*w_norm); wz/w_norm * sin(0.5*w_norm)];

end