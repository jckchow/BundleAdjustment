function [outputQuaternion] = rotationMatrixToQuaternion(R)


quaternion = zeros(1,4);
testTrace  = zeros(1,4);

testTrace(1) = 1.0 + R(1,1) +  R(2,2) +  R(3,3);
testTrace(2) = 1.0 + R(1,1) -  R(2,2) -  R(3,3);
testTrace(3) = 1.0 - R(1,1) +  R(2,2) -  R(3,3);
testTrace(4) = 1.0 - R(1,1) -  R(2,2) +  R(3,3);

[~,I] = max(testTrace);

switch (I)
    
    case 1
        
        quaternion(1) =   0.5 * sqrt( 1.0 + R(1,1) + R(2,2) + R(3,3) );
        
        quaternion(2) =   (R(2,3) - R(3,2)) / (4.0 * quaternion(1));
        quaternion(3) =   (R(3,1) - R(1,3)) / (4.0 * quaternion(1));
        quaternion(4) =   (R(1,2) - R(2,1)) / (4.0 * quaternion(1));
        
        
    case 2
        
        quaternion(2) =   0.5 * sqrt( 1.0 + R(1,1) - R(2,2) - R(3,3) );
        
        quaternion(3) =   (R(1,2) + R(2,1)) / (4.0 * quaternion(2));
        quaternion(4) =   (R(1,3) + R(3,1)) / (4.0 * quaternion(2));
        quaternion(1) =   (R(2,3) - R(3,2)) / (4.0 * quaternion(2));
        
    case 3
        
        quaternion(3) =   0.5 * sqrt( 1.0 - R(1,1) + R(2,2) - R(3,3) );
        
        quaternion(2) =   (R(1,2) + R(2,1)) / (4.0 * quaternion(3));
        quaternion(4) =   (R(2,3) + R(3,2)) / (4.0 * quaternion(3));
        quaternion(1) =   (R(3,1) - R(1,3)) / (4.0 * quaternion(3));
        
    case 4
        
        quaternion(4) =   0.5 * sqrt( 1.0 - R(1,1) - R(2,2) + R(3,3) );
        
        quaternion(2) =   (R(3,1) + R(1,3)) / (4.0 * quaternion(4));
        quaternion(3) =   (R(2,3) + R(3,2)) / (4.0 * quaternion(4));
        quaternion(1) =   (R(1,2) - R(2,1)) / (4.0 * quaternion(4));
        
end

% for consistency
if (quaternion(1) < 0)
    quaternion(1) = -quaternion(1);
    quaternion(2) = -quaternion(2);
    quaternion(3) = -quaternion(3);
    quaternion(4) = -quaternion(4);
end

outputQuaternion = quaternion / norm(quaternion);

end


