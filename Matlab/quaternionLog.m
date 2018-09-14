function [w] = quaternionLog(q)

q_norm = norm(q);
v = q(2:4);
v_norm = norm(v);

ln_q =  (v/v_norm) * acos(q(1)/q_norm);

w = 2 * ln_q';

end