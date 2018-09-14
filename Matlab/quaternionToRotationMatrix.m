function [R] = quaternionToRotationMatrix(quaternion)

    R = zeros(3,3);

    R(1,1) = quaternion(1) * quaternion(1) + quaternion(2) * quaternion(2) - quaternion(3) * quaternion(3) - quaternion(4) * quaternion(4);
    R(1,2) = 2 * ( quaternion(1) * quaternion(4) + quaternion(2) * quaternion(3) );
    R(1,3) = 2 * ( quaternion(2) * quaternion(4) - quaternion(1) * quaternion(3) );

    R(2,1) = 2 * ( quaternion(2) * quaternion(3) - quaternion(1) * quaternion(4) );
    R(2,2) = quaternion(1) * quaternion(1) - quaternion(2) * quaternion(2) + quaternion(3) * quaternion(3) - quaternion(4) * quaternion(4);
    R(2,3) = 2 * ( quaternion(1) * quaternion(2) + quaternion(3) * quaternion(4) );

    R(3,1) = 2 * ( quaternion(1) * quaternion(3) + quaternion(2) * quaternion(4) );
    R(3,2) = 2 * ( quaternion(3) * quaternion(4) - quaternion(1) * quaternion(2) );
    R(3,3) = quaternion(1) * quaternion(1) - quaternion(2) * quaternion(2) - quaternion(3) * quaternion(3) + quaternion(4) * quaternion(4);

end

