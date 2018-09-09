function [worldOrientation,worldLocation,xp,yp,cx,cy,omega,phi,kappa] = dlt(imagePoints,worldPoints)

% convert to m to improve numerical stability
worldPoints = worldPoints ./ 1000;

l = reshape(imagePoints', numel(imagePoints), 1);
w = -l;

A = zeros(length(l), 11);

for n = 1:length(imagePoints(:,1))
    A(2*n-1,1) = worldPoints(n,1);
    A(2*n-1,2) = worldPoints(n,2); 
    A(2*n-1,3) = worldPoints(n,3); 
    A(2*n-1,4) = 1;
    A(2*n-1,5) = 0;
    A(2*n-1,6) = 0;
    A(2*n-1,7) = 0;
    A(2*n-1,8) = 0;
    A(2*n-1,9) = -worldPoints(n,1)*imagePoints(n,1);
    A(2*n-1,10) = -worldPoints(n,2)*imagePoints(n,1); 
    A(2*n-1,11) = -worldPoints(n,3)*imagePoints(n,1); 
    
    A(2*n  ,1) = 0;
    A(2*n  ,2) = 0;
    A(2*n  ,3) = 0;
    A(2*n  ,4) = 0;
    A(2*n  ,5) = worldPoints(n,1);
    A(2*n  ,6) = worldPoints(n,2);
    A(2*n  ,7) = worldPoints(n,3);
    A(2*n  ,8) = 1;
    A(2*n  ,9) = -worldPoints(n,1)*imagePoints(n,2);
    A(2*n  ,10) = -worldPoints(n,2)*imagePoints(n,2);
    A(2*n  ,11) = -worldPoints(n,3)*imagePoints(n,2);

end
N = A'*A;
u = A'*w;

X = -N\u;

%% compute the EOP from the DLT parameters

L=sqrt((X(9))^2+(X(10))^2+(X(11))^2);
xo=(X(1)*X(9)+X(2)*X(10)+X(3)*X(11))/(L^2);
yo=(X(5)*X(9)+X(6)*X(10)+X(7)*X(11))/(L^2);

cx=sqrt(((X(1))^2+(X(2))^2+(X(3))^2)/(L^2)-xo^2);
cy=sqrt(((X(5))^2+(X(6))^2+(X(7))^2)/(L^2)-yo^2);

m31=X(9)/L;
m32=X(10)/L;
m33=X(11)/L;

m11=(xo*m31-X(1)/L)/cx;
m12=(xo*m32-X(2)/L)/cx;
m13=(xo*m33-X(3)/L)/cx;

m21=(yo*m31-X(5)/L)/cy;
m22=(yo*m32-X(6)/L)/cy;
m23=(yo*m33-X(7)/L)/cy;

L_matrix=[X(1) X(2) X(3);
          X(5) X(6) X(7);
          X(9) X(10) X(11)];

% camera's X_c, Y_c, Z_c (m)
% camera_position=-inv(L_matrix)*[X(4); X(8); 1];
camera_position=-L_matrix\[X(4); X(8); 1];

% make it orthonormal again
M = [m11 m12 m13
     m21 m22 m23
     m31 m32 m33];

% [u, s, vt] = svd(M);
% M = u*vt';
% M = orth(M);

q =rotationMatrixToQuaternion(M);

w = quaternionLog(q);
q = quaternionExp(w);

R = quaternionToRotationMatrix(q);


% % % % % % % 
% % % % % % % % M = quat2rotm(quatnormalize(rotm2quat(M)));
% % % % % % % 
% % % % % % % % camera's omega, phi, kappa (degreesm11 = M(1,1);
% % % % % % % m21 = M(2,1);
% % % % % % % m32 = M(3,2);
% % % % % % % m31 = M(3,1);
% % % % % % % m33 = M(3,3);
% % % % % % % 
% % % % % % % omega=atan2(-m32,m33);
% % % % % % % phi=asin(m31);
% % % % % % % kappa=atan2(-m21,m11);
% % % % % % % 
% % % % % % %     m_11 = cos(phi) * cos(kappa) ;
% % % % % % %     m_12 = sin(omega) * sin(phi) * cos(kappa) + cos(omega) * sin(kappa) ;
% % % % % % %     m_13 = -cos(omega) * sin(phi) * cos(kappa) + sin(omega) * sin(kappa) ;
% % % % % % %     m_21 = -cos(phi) * sin(kappa) ;
% % % % % % %     m_22 = -sin(omega) * sin(phi) * sin(kappa) + cos(omega) * cos(kappa) ;
% % % % % % %     m_23 = cos(omega) * sin(phi) * sin(kappa) + sin(omega) * cos(kappa) ;
% % % % % % %     m_31 = sin(phi) ;
% % % % % % %     m_32 = -sin(omega) * cos(phi) ;
% % % % % % %     m_33 = cos(omega) * cos(phi) ;
% % % % % % % 
% % % % % % %     R=[m_11 m_12 m_13;
% % % % % % %        m_21 m_22 m_23;
% % % % % % %        m_31 m_32 m_33];
% % % % % % %    
% % % % % % % e = asin(-M(3,3));
% % % % % % % a = atan2(M(3,1),-M(3,2));
% % % % % % % r = atan2(M(1,3),M(2,3));
% % % % % % % 
% % % % % % % m_11 = cos(r)*cos(a)+sin(r)*sin(e)*sin(a);
% % % % % % % m_12 = cos(r)*sin(a)-sin(r)*sin(e)*cos(a);
% % % % % % % m_13 = sin(r)*cos(e);
% % % % % % % m_21 = -sin(r)*cos(a)+cos(r)*sin(e)*sin(a);
% % % % % % % m_22 = -sin(r)*sin(a)-cos(r)*sin(e)*cos(a);
% % % % % % % m_23 = cos(r)*cos(e);
% % % % % % % m_31 = cos(e)*sin(a);
% % % % % % % m_32 = -cos(e)*cos(a);
% % % % % % % m_33 = -sin(e);
% % % % % % % 
% % % % % % % R1_90 = [1 0 0;
% % % % % % %         0 cos(pi/2) sin(pi/2)
% % % % % % %         0 -sin(pi/2) cos(pi/2)];
% % % % % % % 
% % % % % % %     Q=[m_11 m_12 m_13;
% % % % % % %        m_21 m_22 m_23;
% % % % % % %        m_31 m_32 m_33];
% % % % % % %    
% % % % % % %     R = Q'*M;
% % % % % % % 
% % % % % % %     w = atan2(-R(3,2),R(3,3)) * 180/pi
% % % % % % %     p = asin(R(3,1)) * 180/pi
% % % % % % %     k = atan2(-R(2,1),R(1,1)) * 180/pi
% % % % % % % %     e = asin(-R(3,3)) * 180/pi
% % % % % % % %     a = atan2(R(3,1),-R(3,2)) * 180/pi
% % % % % % % %     r = atan2(R(1,3),R(2,3)) * 180/pi
      
worldOrientation = R;
worldLocation = camera_position';

% convert to back to mm
worldLocation = worldLocation .* 1000;

omega = w(1);
phi = w(2);
kappa = w(3);

xp = xo;
yp = yo;
end

