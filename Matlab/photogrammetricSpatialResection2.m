close all; clear all; clc;

%% Input
pathname = 'C:\Users\jckch\OneDrive - University of Calgary\Google Drive\Omni-Directional Cameras\Data\GoPro\jacky\';
filenameIOP = 'gopro.iop';
filenameXYZ = 'points.xyz';
filenameImage = 'goproLess.pho';

A = [1137.886103	1227.115004	
1140.083816	1631.356924	
2663.355604	1207.455754	
2677.38246	1583.93587	
1900.414702	1156.033243
1377.027515	1136.027502
2459.53647	1738.825767
];

B = [
1144.60804	1231.092912
1149.768555	1635.417309
2674.252056	1199.993218
2691.125526	1577.790074  
1907.528829	1154.005565
1382.77957	1138.149118
2473.195773	1734.771435
];
   
C=[
178.605665	1044.007709
161.647866	1679.151792
2722.72344	990.541119	
2733.473067	1742.579286
1202.668513	864.904311
459.017446	881.57009
2276.73964	2045.734056
    ];

D =[
146.24905	1040.878986	
132.74513	1680.903128	
2694.515176	977.869366	
2706.454555	1729.511957	
1174.273463	856.311957  
427.959999	875.95003
2251.74885	2034.180583
];

E =[
    626.060458	925.435748	
646.662535	1616.663033
3379.551838	850.487979	
3389.673992	1559.647141
1942.040198	675.782326
976.17866	720.338788
3014.051252	1854.130356	
    ];

F =[
    638.214851	944.483729	
658.925787	1633.129939
3386.026362	856.123189	
3400.448334	1566.637622
1948.730144	691.948146
986.377805	739.507181
3025.022421	1864.132133
];

G=[
    302.887125	1437.615521
83.277036	1943.796757
2397.151312	1409.990864
2508.074755	2050.612477
1151.182414	1365.505608
571.762837	1342.463527
2049.0707	2381.606709
    ];

H=[
  259.232147	1416.92353	
49.343229	1932.741312
2358.478465	1366.886663
2472.55412	2004.608616
1112.93018	1331.012563
529.825021	1315.967633
2020.488253	2336.744958
];

I=[
587.152517	1566.404215
346.675786	2156.779515
2943.568516	1533.212594
3160.530069	2101.396421
1712.982046	1507.915465
935.693636	1469.058728
2855.041882	2459.981669	

    ];

J=[
    1150.718162	1570.714336
958.407315	2265.862753
3531.776252	1488.924227
3811.055157	1988.455295
2518.486212	1479.048955
1601.37903	1454.897609
3655.106148	2314.200017
];

K=[
   266.915119	449.360547	
477.079293	1025.212216
3031.735313	336.834091	
2863.231831	1042.987862
 1367.645708	115.35396
 510.924678	223.618892
 2422.567022	1225.982299
];

L=[
457.551557	569.680039	
649.086396	1236.719078
3331.409142	572.498314	
3169.96883	1197.890091
1893.042787	278.205187
817.487215	321.813566
2800.259167	1395.324776
    ];

imagePoints= cat(3,A,B,C,D,E,F,G,H,I,K,L);
        
worldPoints = [
-3768.344725	822.365959	-1893.014451
-3774.809493	816.866694	-3083.603062
-57.145906	3174.333979	-1962.557558	
-49.096474	3180.294148	-3015.075855	
-1858.642579	2031.215616	-1798.970304
-3112.91497	1236.420107	-1672.9819
-594.331944	2834.199694	-3414.797937
];

    [coeff, score, roots] = pca(worldPoints);
    
    [coeff,score,roots] = pca(worldPoints);
    basis = coeff(:,1:2);
    % normal vector of the plane A, B, and C
    normal = coeff(:,3);
    pctExplained = roots' ./ sum(roots);
    
    [n,p] = size(worldPoints);
    meanX = mean(worldPoints,1);
    distance=meanX*normal;
    
    % Equation of a plane AX+BY+CZ-D=0
    ABCD=[normal; distance];
    
% Remove the X component
kappa=atan2(ABCD(1,1),ABCD(2,1));

R3=[cos(kappa) sin(kappa) 0;
    -sin(kappa) cos(kappa) 0;
    0 0 1];


ABCD_X(1:3,1)=R3'*ABCD(1:3);

for i=1:length(worldPoints(:,1))
    worldPoints(i,1:3)=(R3'*(worldPoints(i,1:3))')';
end

% Remove the Y component
omega=atan2(ABCD_X(2,1),ABCD_X(3,1));

R1=[1 0 0;
    0 cos(omega) sin(omega);
    0 -sin(omega) cos(omega)];

ABCD_X(1:3,1)=R1'*ABCD_X(1:3);

for i=1:length(worldPoints(:,1))
    worldPoints(i,1:3)=(R1'*(worldPoints(i,1:3))')';
end

circle_Z=ABCD_X(3,1).*ABCD(4,1);

worldPoints = worldPoints(:,1:2);

%% read and process the files
in=fopen([pathname, filenameIOP],'r');
data=textscan(in,'%d %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f','headerlines',0);
fclose(in);

cx = data{9};
cy = data{9};
xp1 = data{7};
yp1 = data{8};
xp = 0;
yp = 0;

K = [cx 0 0
     0 cy 0
     xp yp 1];
% cameraParams = cameraParameters('IntrinsicMatrix',intrinsics.IntrinsicMatrix);
cameraParams = cameraParameters('IntrinsicMatrix',K);

% attempt to calibrate fisheye camera
cameraParams = estimateCameraParameters(imagePoints, worldPoints, ...
                                      'ImageSize', [3840, 2880], 'NumRadialDistortionCoefficients', 3,'EstimateTangentialDistortion',false);
% [cameraParams, imagesUsed, estimationErrors] = estimateFisheyeParameters(imagePoints, worldPoints, [2880,3840]);

% Visualize calibration accuracy.
figure
showReprojectionErrors(cameraParams);

% Visualize camera extrinsics.
figure
showExtrinsics(cameraParams);
drawnow
                                  
I = imread('C:\Users\jckch\OneDrive - University of Calgary\Google Drive\Omni-Directional Cameras\Data\Photos\GOPR0365.JPG');
J1 = undistortImage(I,cameraParams,'OutputView','full');
% J1 = undistortFisheyeImage(I,cameraParams.Intrinsics);

figure
imshowpair(I, J1, 'montage')
title('Original Image (left) vs. Corrected Image (right)')

% J2 = undistortFisheyeImage(I, cameraParams.Intrinsics, 'OutputView', 'full');
% figure
% imshow(J2)
% title('Full Output View')

save cameraParams
load cameraParams
% read object space
in=fopen([pathname, filenameXYZ],'r');
data=textscan(in,'%d %f %f %f %f %f %f','Delimiter', ' ', 'headerlines',0);
fclose(in);
ID_obj = data{1};
XYZ = [data{2}, data{3} data{4}];

% image points
in=fopen([pathname, filenameImage],'r');
data=textscan(in,'%d %d %f %f %f %f %f %f', 'Delimiter', ' ','headerlines',0);
fclose(in);

ID_img = data{1};
ID_EOP = data{2};
% img = [data{3}-xp1, data{4}+yp1];
img = [data{3}, data{4}];




ID_EOP_unique = unique(ID_EOP);

figure; plot(img(:,1),img(:,2),'b.')

%% Estimate EOPs
figure
pcshow(XYZ,'VerticalAxis','Y','VerticalAxisDir','down', ...
     'MarkerSize',100);
hold on
plot3(Xo_vector,Yo_vector,Zo_vector,'r*')
axis equal
xlabel('X')
ylabel('Y')
zlabel('Z')

w_vector = [];
p_vector = [];
k_vector = [];
Xo_vector = [];
Yo_vector = [];
Zo_vector = [];
EOPID_vector = [];

missing_obj = [];
error_vector = [];
for n = 1:1:length(ID_EOP_unique)
    disp(n)
    I = find(ID_EOP == ID_EOP_unique(n)); % I gives you all the image measurements from that station
    tempImagePoints = img(I,:); % get all the image measurements from that station
    tempID = ID_img(I,:);
    
    imagePoints = [];
    worldPoints = [];
    EOPID_vector = [EOPID_vector; n];
    for m = 1:length(tempID)
        J = find(ID_obj == tempID(m));
        if isempty(J) % if we don't have the obj space coordinate of that image point
%             imagePoints(m,:) = [];
            missing_obj = [missing_obj; tempID(m)]; % use this to remove points in .pho
        else
            imagePoints = [imagePoints; tempImagePoints(m,:)];
            worldPoints = [worldPoints; XYZ(J,:)];
%             imagePoints = [imagePoints; tempID(m)];
%             worldPoints = [worldPoints; ID_obj(J,:)];


        end
    end
    
%     [worldOrientation,worldLocation,inlierIdx,status] = estimateWorldCameraPose(imagePoints,worldPoints,cameraParams);

    imagePoints_original = imagePoints;
    worldPoints_original = worldPoints;
    dist = sqrt((imagePoints(:,1)-cameraParams.PrincipalPoint(1)).^2 + (imagePoints(:,2)-cameraParams.PrincipalPoint(2)).^2);
    [dist_sorted,K] = sort(dist);
    KK = find(K <= 25);
    
%     KK = find (dist < 10000000);
    
%     figure;
%     plot(imagePoints(:,1),imagePoints(:,2),'m.')

    imagePoints = imagePoints(KK,:);
    worldPoints = worldPoints(KK,:);
    
    [worldOrientation,worldLocation,inlierIdx,status] = estimateWorldCameraPose(imagePoints,worldPoints,cameraParams, 'MaxNumTrials', 50000, 'MaxReprojectionError', 2);
    disp(worldLocation)
    %     [worldOrientation1,worldLocation1,xp,yp,c_x,c_y,omega,phi,kappa] = dlt([imagePoints(:,1),-imagePoints(:,2)],worldPoints);
% % % % 
% % % %     m_11 = cos(phi) * cos(kappa) ;
% % % %     m_12 = sin(omega) * sin(phi) * cos(kappa) + cos(omega) * sin(kappa) ;
% % % %     m_13 = -cos(omega) * sin(phi) * cos(kappa) + sin(omega) * sin(kappa) ;
% % % %     m_21 = -cos(phi) * sin(kappa) ;
% % % %     m_22 = -sin(omega) * sin(phi) * sin(kappa) + cos(omega) * cos(kappa) ;
% % % %     m_23 = cos(omega) * sin(phi) * sin(kappa) + sin(omega) * cos(kappa) ;
% % % %     m_31 = sin(phi) ;
% % % %     m_32 = -sin(omega) * cos(phi) ;
% % % %     m_33 = cos(omega) * cos(phi) ;
% % % % 
% % % %     M=[m_11 m_12 m_13;
% % % %        m_21 m_22 m_23;
% % % %        m_31 m_32 m_33];
% % % %    
% % % %     M1=M;

    q = rotationMatrixToQuaternion(worldOrientation);
    w = quaternionLog(q);
    
    q = quaternionExp(w);
%     q = quaternionExp([omega,phi,kappa]);
    M1 = quaternionToRotationMatrix(q);
    
    omega = atan2(-M1(3,2),M1(3,3));
    phi = asin(M1(3,1));
    kappa = atan2(-M1(2,1),M1(1,1));
    
    m_11 = cos(phi) * cos(kappa) ;
    m_12 = sin(omega) * sin(phi) * cos(kappa) + cos(omega) * sin(kappa) ;
    m_13 = -cos(omega) * sin(phi) * cos(kappa) + sin(omega) * sin(kappa) ;
    m_21 = -cos(phi) * sin(kappa) ;
    m_22 = -sin(omega) * sin(phi) * sin(kappa) + cos(omega) * cos(kappa) ;
    m_23 = cos(omega) * sin(phi) * sin(kappa) + sin(omega) * cos(kappa) ;
    m_31 = sin(phi) ;
    m_32 = -sin(omega) * cos(phi) ;
    m_33 = cos(omega) * cos(phi) ;

    M=[m_11 m_12 m_13;
       m_21 m_22 m_23;
       m_31 m_32 m_33];
    
    omega = pi;
    phi   = 0;
    kappa = 0;

    m_11 = cos(phi) * cos(kappa) ;
    m_12 = sin(omega) * sin(phi) * cos(kappa) + cos(omega) * sin(kappa) ;
    m_13 = -cos(omega) * sin(phi) * cos(kappa) + sin(omega) * sin(kappa) ;
    m_21 = -cos(phi) * sin(kappa) ;
    m_22 = -sin(omega) * sin(phi) * sin(kappa) + cos(omega) * cos(kappa) ;
    m_23 = cos(omega) * sin(phi) * sin(kappa) + sin(omega) * cos(kappa) ;
    m_31 = sin(phi) ;
    m_32 = -sin(omega) * cos(phi) ;
    m_33 = cos(omega) * cos(phi) ;

    M=[m_11 m_12 m_13;
       m_21 m_22 m_23;
       m_31 m_32 m_33];

    R = M * worldOrientation;
    T = worldLocation;
% %     
% %     error_x = 0;
% %     error_y = 0;
% %     dx = [];
% %     dy = [];
% %     for m = 1:length(worldPoints_original(:,1))
% %         temp = R * (worldPoints_original(m,:)' - worldLocation');
% %         x_img = -cx * temp(1) / temp(3);
% %         y_img = -cy * temp(2) / temp(3);
% %         error_x = error_x + (x_img - imagePoints_original(m,1))^2;
% %         error_y = error_y + (y_img + imagePoints_original(m,2))^2;
% %         
% %         dx = [dx, x_img - imagePoints_original(m,1)];
% %         dy = [dy, y_img + imagePoints_original(m,2)];
% %     end
% %     error = error_x + error_y;
% %     e = sqrt(error / (2*length(worldPoints_original(:,1))));
% %     error_vector = [error_vector; e];
% % 
% %     figure
% %     quiver(imagePoints_original(:,1), imagePoints_original(:,2), dx', dy','r')
% %     
% %     error_x = 0;
% %     error_y = 0;
% %     dx = [];
% %     dy = [];
% %     for m = 1:length(worldPoints_original(:,1))
% %         temp = worldOrientation1 * (worldPoints_original(m,:)' - worldLocation1');
% %         x_img = xp -c_x * temp(1) / temp(3);
% %         y_img = yp -c_y * temp(2) / temp(3);
% %         error_x = error_x + (x_img - imagePoints_original(m,1))^2;
% %         error_y = error_y + (y_img + imagePoints_original(m,2))^2;
% %         
% %         dx = [dx, x_img - imagePoints_original(m,1)];
% %         dy = [dy, y_img + imagePoints_original(m,2)];
% %     end
% %     error = error_x + error_y;
% %     e = sqrt(error / (2*length(worldPoints_original(:,1))));
% %     error_vector = [error_vector; e];
% %     
% %     figure
% %     quiver(imagePoints_original(:,1), imagePoints_original(:,2), dx', dy')
% %        
% % % % % %     figure
% % % % % %     pcshow(worldPoints,'VerticalAxis','Y','VerticalAxisDir','down', ...hg
% % % % % %          'MarkerSize',100);
% % % % % %     hold on  
% % % % % % %     plotCamera('Size',100,'Orientation',worldOrientation,'Location',...
% % % % % % %          worldLocation);
% % % % % %     plotCamera('Size',100,'Orientation',M1,'Location',...
% % % % % %          worldLocation1);
% % % % % %     hold off
% % % % % %     xlabel('X')
% % % % % %     ylabel('Y')
% % % % % %     zlabel('Z')
    
    w = atan2(-R(3,2),R(3,3));
    p = asin(R(3,1));
    k = atan2(-R(2,1),R(1,1));
    
    Xo_vector = [Xo_vector; T(1)];
    Yo_vector = [Yo_vector; T(2)];
    Zo_vector = [Zo_vector; T(3)];
    
    w_vector = [w_vector; w];
    p_vector = [p_vector; p];
    k_vector = [k_vector; k];
    
%     % Doing a random initialization
%     tic
%     pose_vector = [];
%     error_vector = [];
%     for yaw = -90:90:90
%         for roll = -90:20:90
%            for pitch = -180:20:170
%                for Tx = min(XYZ(:,1)):500:max(XYZ(:,1))
%                    for Ty = min(XYZ(:,2)):500:max(XYZ(:,2))
%                        for Tz = min(XYZ(:,3)):500:max(XYZ(:,3))
%                             omega = roll*pi/180;
%                             phi   = pitch*pi/180;
%                             kappa = yaw*pi/180;
% 
%                             m_11 = cos(phi) * cos(kappa) ;
%                             m_12 = sin(omega) * sin(phi) * cos(kappa) + cos(omega) * sin(kappa) ;
%                             m_13 = -cos(omega) * sin(phi) * cos(kappa) + sin(omega) * sin(kappa) ;
%                             m_21 = -cos(phi) * sin(kappa) ;
%                             m_22 = -sin(omega) * sin(phi) * sin(kappa) + cos(omega) * cos(kappa) ;
%                             m_23 = cos(omega) * sin(phi) * sin(kappa) + sin(omega) * cos(kappa) ;
%                             m_31 = sin(phi) ;
%                             m_32 = -sin(omega) * cos(phi) ;
%                             m_33 = cos(omega) * cos(phi) ;
% 
%                             M=[m_11 m_12 m_13;
%                                m_21 m_22 m_23;
%                                m_31 m_32 m_33];
%                            
%                            pose_vector = [pose_vector; [roll, pitch, yaw, Tx, Ty, Tz]];
%                                 error_x = 0;
%                                 error_y = 0;
%                                 for m = 1:length(worldPoints(:,1))
%                                     temp = M * (worldPoints(m,:)' - [Tx, Ty, Tz]');
%                                     x_img = -cx * temp(1) / temp(3);
%                                     y_img = -cy * temp(2) / temp(3);
%                                     error_x = error_x + x_img - imagePoints(m,1);
%                                     error_y = error_y + y_img + imagePoints(m,2);
%                                 end
%                                 error = sqrt(error_x^2 + error_y^2);
%                                 e = error / (2*length(worldPoints(:,1)));
%                                 error_vector = [error_vector; e];
%                        end
%                    end
%                end
%            end
%         end
%     end
%     toc
    
    if ~isnan(T(1))
%     pcshow(worldPoints,'VerticalAxis','Y','VerticalAxisDir','down', ...
%          'MarkerSize',100);
%     hold on  
    plotCamera('Size',100,'Orientation',worldOrientation,'Location',...
         worldLocation);
%     hold off
    end

end

cx
nanmean(error_vector)
nanmedian(error_vector)

hold off
axis equal
xlabel('X')
ylabel('Y')
zlabel('Z')

% disp('Recovered R')
% R

% disp('Difference in R')
% R_diff = R_true' * R

% w = atan2(-R_diff(3,2),R_diff(3,3));
% p = asin(R_diff(3,1));
% k = atan2(-R_diff(2,1),R_diff(1,1));

% disp('Difference in OPK')
% [w,p,k]

% Xo_vector = round(Xo_vector,1);
% Yo_vector = round(Yo_vector,1);
% Zo_vector = round(Zo_vector,1);
% 
% w_vector = round(w_vector,2);
% p_vector = round(p_vector,2);
% k_vector = round(k_vector,2);


% OPK = [mode(w_vector),mode(p_vector),mode(k_vector)];
% disp('Recovered T')
% T = [mode(Xo_vector), mode(Yo_vector), mode(Zo_vector)]


OPKXYZ = [EOPID_vector, ones(length(EOPID_vector),1)*1, Xo_vector, Yo_vector, Zo_vector, w_vector, p_vector, k_vector];

disp('Writing to file Xo, Yo, Zo (mm), w, p, k(deg)')
out=fopen([pathname,'gopro.eop'],'w');
for n = 1:length(OPKXYZ(:,1))
    fprintf(out, '%d \t %d \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', OPKXYZ(n,1),OPKXYZ(n,2),OPKXYZ(n,3),OPKXYZ(n,4),OPKXYZ(n,5),OPKXYZ(n,6)*180/pi,OPKXYZ(n,7)*180/pi,OPKXYZ(n,8)*180/pi);
end
fclose(out);

missing_obj = unique(missing_obj)

disp("Success ^-^")
