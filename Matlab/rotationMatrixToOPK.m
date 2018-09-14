function [omega, phi, kappa] = rotationMatrixToOPK(R)


    % convert the rotation matrix to omega, phi, kapp
    omega = atan2(-R(3,2),R(3,3));
    phi   = asin(R(3,1));
    kappa = atan2(-R(2,1),R(1,1));

end


