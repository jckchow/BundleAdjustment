%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Program: Single photo resection 
%%% Programmer: Jacky Chow
%%% Date: March 25, 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clear all; clc;
%% Input

% pathname = 'C:\Users\jckch\OneDrive - University of Calgary\Google Drive\Omni-Directional Cameras\Data\NikonDSLR\jacky_2020_03_22\';
% filenameIOP = 'nikon.iop'; % treat as constant
% filenameXYZ = 'nikon.xyz'; % treat as constant
% % filenameXYZ = 'nikonLowWeight_centred.xyz'; % treat as constant
% filenameImage = 'nikon.pho'; % outliers will be removed
% outputEOPFilename = 'nikon.eop';
% outputPhoFilename = 'nikon_screened.pho'; % without outliers based on single photo resection
% outputXYZFilename = 'nikon_centred.xyz';

pathname = 'C:\Users\jckch\OneDrive - University of Calgary\Google Drive\Omni-Directional Cameras\Data\GoPro\jacky_2020_03_31\';
filenameIOP = 'gopro.iop'; % treat as constant
% filenameXYZ = 'goproLowWeight_centred.xyz'; % treat as constant
filenameXYZ = 'gopro.xyz'; % treat as constant
filenameImage = 'gopro.pho'; % outliers will be removed
outputEOPFilename = 'gopro.eop';
outputPhoFilename = 'gopro_screened.pho'; % without outliers based on single photo resection RANSAC
outputXYZFilename = 'gopro_centred.xyz';

% 1 = collinearity, 2 = stereographic projection model
mode = 1;

numRANSACIter = 1000; % number of RANSAC iterations to do
KNN = 30; % using the # of img points nearest the principal point, assumed to be least effected by distortions
%% read and process the files

% Read in the IOP file
in=fopen([pathname, filenameIOP],'r');
data=textscan(in,'%d %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f','headerlines',0);
fclose(in);

cx = data{9};
cy = data{9};
xp = data{7};
yp = data{8};
xp0 = 0;
yp0 = 0;
K = [cx 0 0
     0 cy 0
     xp0 yp0 1]; %will apply xp and yp manually, hence it is zero in the K matrix
% cameraParams = cameraParameters('IntrinsicMatrix',intrinsics.IntrinsicMatrix);
cameraParams = cameraParameters('IntrinsicMatrix',K);

% read in the object space file
in=fopen([pathname, filenameXYZ],'r');
data=textscan(in,'%d %f %f %f %f %f %f','headerlines',0);
fclose(in);

ID_XYZ = data{1};
XYZ = [data{2}, data{3} data{4}];
XYZ_stdDev = [data{5}, data{6} data{7}];
% XYZ(:,1) = XYZ(:,1) - mean(XYZ(:,1)); % reduce to centroid
% XYZ(:,2) = XYZ(:,2) - mean(XYZ(:,2));
% XYZ(:,3) = XYZ(:,3) - mean(XYZ(:,3));


% read in the image points
in=fopen([pathname, filenameImage],'r');
data=textscan(in,'%d %d %f %f %f %f %f %f','headerlines',0);
fclose(in);

ID_img = data{1};
ID_EOP = data{2};
img = [data{3}-xp, data{4}+yp]; % openCV's row and column coordinate system, but now centred
img_stdDev = [data{5}, data{6}];
img_corr = [data{7}, data{8}];

ID_EOP_unique = unique(ID_EOP);

figure;
plot(img(:,1),-img(:,2),'b.')
title('Image Points')

%% Estimate EOPs
figure
pcshow(XYZ,'VerticalAxis','Y','VerticalAxisDir','down', ...
     'MarkerSize',100);
hold on
% axis equal
% xlabel('X')
% ylabel('Y')
% zlabel('Z')

w_vector = [];
p_vector = [];
k_vector = [];
Xo_vector = [];
Yo_vector = [];
Zo_vector = [];
EOPID_vector = [];

% detect missing object points
missing_obj = [];
error_vector_mean_before = zeros(length(ID_EOP_unique),1);
error_vector_max_before = zeros(length(ID_EOP_unique),1);
error_vector_mean_after = zeros(length(ID_EOP_unique),1);
error_vector_max_after = zeros(length(ID_EOP_unique),1);

outlierIndex = false(length(ID_img),1);
reprojectionError_original = zeros(length(ID_img),1);
reprojectionError_original2 = zeros(length(ID_img),2);

if (mode == 1)
    disp('Using conventional collinearity equations for single photo resection')
end

if (mode == 2)
    disp('Using stereographic projection collinearity equations for single photo resection')
end
for n = 1:1:length(ID_EOP_unique)
    disp(['Image: ', num2str(n),' out of ', num2str(length(ID_EOP_unique))])
    I = find(ID_EOP == ID_EOP_unique(n)); % I gives you all the image measurements from that station
    perStationIndex = I;
    tempImagePoints = img(I,:); % get all the image measurements from that station
    tempID = ID_img(I,:);
    
    imagePointsID = [];
    imagePoints = [];
    worldPointsID = [];
    worldPoints = [];
    worldPointsStdDev = [];
    perStation = [];
    EOPID_vector = [EOPID_vector; n];
    for m = 1:length(tempID)
        J = find(ID_XYZ == tempID(m));
        if isempty(J) % if we don't have the obj space coordinate of that image point
%             imagePoints(m,:) = [];
            disp('Missing object space coordinates for this image point, so removing it from further processing...')
            disp(['   Point ID: ' , num2str(tempID(m))]);
            missing_obj = [missing_obj; tempID(m)]; % use this to remove points in .pho
        else
            imagePointsID = [imagePointsID; tempID(m)];
            imagePoints = [imagePoints; tempImagePoints(m,:)];
            worldPointsID = [worldPointsID; tempID(m)];
            worldPoints = [worldPoints; XYZ(J,:)];
            worldPointsStdDev = [worldPointsStdDev; XYZ_stdDev(J,:)];
            perStation = [perStation; perStationIndex(m)];
        end
    end
    
    disp(['   Number of original image points: ', num2str(length(tempImagePoints))])
    disp(['   Number of homologous points found: ', num2str(length(imagePoints))])
%     [worldOrientation1,worldLocation1,inlierIdx,status] = estimateWorldCameraPose(imagePoints,worldPoints,cameraParams);

% filter the points by proximity to middle of image
    ID_img_original = ID_img;
    ID_EOP_original = ID_EOP;
    imagePoints_original = imagePoints; % all image points even not filtered
    worldPoints_original = worldPoints;
    dist = sqrt(imagePoints(:,1).^2 + (-imagePoints(:,2)).^2);
%     figure;
%     plot((imagePoints(:,1)-xp), (-(imagePoints(:,2)+yp)),'b.')
    [dist_sorted,K] = sort(dist);
%     KK = find(K <= KNN); % use only the K points close to the middle of the image
    KK = K(1:KNN);

    imagePoints = imagePoints(KK,:);
    worldPoints = worldPoints(KK,:);
    imagePointsID = imagePointsID(KK,:);
    worldPointsID = worldPointsID(KK,:);
    disp(['      Number of distance filtered centre points: ', num2str(length(imagePoints(:,1)))])
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Do ransac approach to find outliers in image measurements
    rng(0,'twister');
        indexInliers = logical(length(imagePoints));
        inlierThreshold = 10;
        mostInliers = [];
        mostInliersErrors = 1E10;
        disp(['      Starting RANSAC...'])
        [worldOrientation,worldLocation,inlierIdx,status] = estimateWorldCameraPose(imagePoints,worldPoints,cameraParams,'MaxNumTrials', 10000, 'MaxReprojectionError', 5);  

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

            w = atan2(-R(3,2),R(3,3));
            p = asin(R(3,1));
            k = atan2(-R(2,1),R(1,1));
            
            if (mode == 1)
                x0 = [w;p;k;T(1);T(2);T(3)]; % initial unknown vector
            end
            
            if (mode == 2)
                radiusInitial = 1E-6;
                x0 = [w;p;k;T(1);T(2);T(3);radiusInitial]; % initial unknown vector
            end

        tic;
        m = 4; % min number of points
        for ransacIter = 1:numRANSACIter

            r = randperm(length(imagePoints(:,1)));
            inliers = r(1:m);
           
            lx = imagePoints(inliers,1);
            ly = imagePoints(inliers,2);
            X = worldPoints(inliers,1);
            Y = worldPoints(inliers,2);
            Z = worldPoints(inliers,3);
            options = optimoptions(@lsqnonlin,'Display', 'off');
            
            if (mode == 1)
                [x,resnorm,residuals,exitflag,output] = lsqnonlin(@(param) collinearityResection(param, cx, cy, X, Y, Z, lx, ly), x0, [], [], options);
            end
            
            if (mode == 2)
                lb = [-1E10; -1E10; -1E10; -1E10; -1E10; -1E10; 1E-9];
                [x,resnorm,residuals,exitflag,output] = lsqnonlin(@(param) stereographicResection(param, cx, cy, X, Y, Z, lx, ly), x0, lb, [], options);
            end
%             x0 = [x0; cx];
%             m = 4;
%             inliers = r(1:m);
%            
%             lx = imagePoints(inliers,1);
%             ly = imagePoints(inliers,2);
%             X = worldPoints(inliers,1);
%             Y = worldPoints(inliers,2);
%             Z = worldPoints(inliers,3);
%             options = optimoptions(@lsqnonlin,'Display', 'off');
%             [x,resnorm,residuals,exitflag,output] = lsqnonlin(@(param) SinglePhotoCalibration(param, X, Y, Z, lx, ly), x0, [], [], options);

%             apostStdDev = sqrt(resnorm / (2*length(lx)));

            % Grow my inliers
                inliers = r;
                lx = imagePoints(inliers,1);
                ly = imagePoints(inliers,2);
                X = worldPoints(inliers,1);
                Y = worldPoints(inliers,2);
                Z = worldPoints(inliers,3);

                if (mode == 1)
                    v = collinearityResection(x, cx, cy, X, Y, Z, lx, ly);
                end
                
                if (mode == 2)
                    v = stereographicResection(x, cx, cy, X, Y, Z, lx, ly);
                end
                
                residuals = reshape(v,length(X),2);
                
                I = find(abs(residuals(:,1))<inlierThreshold);
                J = find(abs(residuals(I,2))<inlierThreshold);
                inliers = I(J);
               
                testError = sqrt(v(inliers)'*v(inliers)) / length(inliers); % this is the residual norm for all
                if(length(inliers) > length(mostInliers))
                    mostInliersErrors = testError;
                    mostInliers = r(inliers);
                elseif length(inliers) == length(mostInliers)
                    if (testError < mostInliersErrors)
                        mostInliersErrors = testError;
                        mostInliers = r(inliers);
                    end
                end
                
%                 figure;
%                 plot(imagePointsID(r), residuals(:,1),'r.')
%                 hold on;
%                 plot(imagePointsID(r), residuals(:,2),'g.')
%                 hold off;
%                 xlabel('Target ID#')
% 
%                 figure;
%                 plot(sqrt(imagePoints(r,1).^2+imagePoints(r,2).^2), residuals(:,1),'r.')
%                 hold on;
%                 plot(sqrt(imagePoints(r,1).^2+imagePoints(r,2).^2), residuals(:,2),'g.')
%                 hold off;
%                 xlabel('Radial Distance')
                
            
%             for growingInliers = m+1:length(r) 
%                 inliers = r(growingInliers);
%                 lx = imagePoints(inliers,1);
%                 ly = imagePoints(inliers,2);
%                 X = worldPoints(inliers,1);
%                 Y = worldPoints(inliers,2);
%                 Z = worldPoints(inliers,3);
% 
%                 v = f(x, cx, cy, X, Y, Z, lx, ly);
% 
%                 e = norm(v);
% 
%                 if (abs(v(1))<inlierThreshold && abs(v(2))<inlierThreshold)
%                     if(growingInliers > length(mostInliers))
%                         mostInliers = r(1:growingInliers);
%                     end
%                 else
%                     break;
%                 end
%             end
            
        end

        disp(['      Number of RANSAC inliers for estimating EOP: ', num2str(length(mostInliers))])
        toc

        % Run one more estimation with most inliers again to update unknown parameters
        
        inliers = mostInliers;
        lx = imagePoints(inliers,1);
        ly = imagePoints(inliers,2);
        X = worldPoints(inliers,1);
        Y = worldPoints(inliers,2);
        Z = worldPoints(inliers,3);
        options = optimoptions(@lsqnonlin,'Display', 'off');
        if (mode == 1)
            [x,resnorm,residuals,exitflag,output] = lsqnonlin(@(param) collinearityResection(param, cx, cy, X, Y, Z, lx, ly), x0, [], [], options);
        end
        
        if (mode == 2)
            [x,resnorm,residuals,exitflag,output] = lsqnonlin(@(param) stereographicResection(param, cx, cy, X, Y, Z, lx, ly), x0, [], [], options);            
        end
            
        apostStdDev = sqrt(resnorm / (2*length(lx))); 
%         apostStdDev = residuals * img_stdDev';

        disp(['         Reprojection error of inliers: ', num2str(apostStdDev)])

% % % % % %         % Calculate the RANSAC Inlier error statistics
% % % % % %         e_threshold = mean(residuals)+ 3.091*std(residuals); %99.9 confidence
% % % % % %      

        % estimate residuals for everything
        lx = imagePoints_original(:,1);
        ly = imagePoints_original(:,2);
        X = worldPoints_original(:,1);
        Y = worldPoints_original(:,2);
        Z = worldPoints_original(:,3);

        if (mode == 1)
            residuals = collinearityResection(x, cx, cy, X, Y, Z, lx, ly);
        end
        
        if (mode == 2)
            residuals = stereographicResection(x, cx, cy, X, Y, Z, lx, ly);            
        end
        
        residuals = reshape(residuals,length(X),2);
        
        reprojectionError_original2(perStation,:) = residuals;
        
%         figure;
%         plot(tempID, residuals(:,1),'b')
%         hold on;
%         plot(tempID, residuals(:,2),'r')
%         hold off;
%         xlabel('Target ID#')
%         
%         figure;
%         plot(sqrt(lx.^2+ly.^2), residuals(:,1),'r.')
%         hold on;
%         plot(sqrt(lx.^2+ly.^2), residuals(:,2),'g.')
%         hold off;
%         xlabel('Radial distance')

% % % % %         
% % % % %         x0 = [x0; cx];
% % % % %         [x,resnorm,residuals,exitflag,output] = lsqnonlin(@(param) SinglePhotoCalibration(param, X, Y, Z, lx, ly), x0, [], [], options);
% % % % %         
% % % % %         % estimate residuals for everything
% % % % %         lx = imagePoints_original(:,1);
% % % % %         ly = imagePoints_original(:,2);
% % % % %         X = worldPoints_original(:,1);
% % % % %         Y = worldPoints_original(:,2);
% % % % %         Z = worldPoints_original(:,3);
% % % % % 
% % % % %         residuals = SinglePhotoCalibration(x, X, Y, Z, lx, ly);
% % % % %         
% % % % %         residuals = reshape(residuals,length(X),2);
% % % % %         
% % % % %         reprojectionError_original2(perStation,:) = residuals;
% % % % %         
% % % % %         figure;
% % % % %         plot(imagePointsID, residuals(:,1),'b.')
% % % % %         hold on;
% % % % %         plot(imagePointsID, residuals(:,2),'r.')
% % % % %         hold off;   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % % % % % %     [worldOrientation,worldLocation,inlierIdx,status] = estimateWorldCameraPose(imagePoints,worldPoints,cameraParams,'MaxNumTrials', 10000, 'MaxReprojectionError', 5);  
% % % % % % % % % %         [worldOrientation1,worldLocation1,xp,yp,cx,cy,omega,phi,kappa] = dlt([imagePoints(:,1),-imagePoints(:,2)],worldPoints(:,:));
% % % % % % % % % 
% % % % % % % % %         % Calculate the RANSAC Inlier error statistics
% % % % % % % % %         dx = zeros(length(worldPoints),1);
% % % % % % % % %         dy = zeros(length(worldPoints),1);
% % % % % % % % %         for m = 1:length(worldPoints(:,1))
% % % % % % % % %     %         temp = R * (worldPoints_original(m,:)' - worldLocation');
% % % % % % % % %             temp = worldOrientation * (worldPoints(m,:)' - worldLocation');
% % % % % % % % %             x_img = cx * temp(1) / temp(3);
% % % % % % % % %             y_img = cy * temp(2) / temp(3);
% % % % % % % % % 
% % % % % % % % %             % this calculation down in OpenCV coordinates but centered
% % % % % % % % %             dx(m) = x_img - imagePoints(m,1); %this points to the right
% % % % % % % % %             dy(m) = y_img - imagePoints(m,2); %this points down
% % % % % % % % %         end
% % % % % % % % % 
% % % % % % % % %         e = sqrt(dx.^2 + dy.^2);
% % % % % % % % %         e_threshold = mean(e)+ 3.091*std(e); %99.9 confidence
% % % % % % % % % 
% % % % % % % % % %     figure;
% % % % % % % % % %     plot(1:length(e), e,'b')
% % % % % % % % % %     hold on;
% % % % % % % % % %     plot(1:length(e), e_threshold.*ones(1,length(e)),'r')
% % % % % % % % % %     hold off;
% % % % % % % % %     
% % % % % % % % % %     [worldOrientation,worldLocation,~,~] = estimateWorldCameraPose(imagePoints(mostInliers,:),worldPoints(mostInliers,:),cameraParams,'MaxNumTrials', 10000, 'MaxReprojectionError', 30);
% % % % % % % % %     % calculate the error vector from the estimated EOP for ALL the points,
% % % % % % % % %     % even ones that are at the periphery of the image
% % % % % % % % %     dx = zeros(length(worldPoints_original(:,1)),1);
% % % % % % % % %     dy = zeros(length(worldPoints_original(:,1)),1);
% % % % % % % % %     for m = 1:length(worldPoints_original(:,1))
% % % % % % % % % %         temp = R * (worldPoints_original(m,:)' - worldLocation');
% % % % % % % % %         temp = worldOrientation * (worldPoints_original(m,:)' - worldLocation');
% % % % % % % % %         x_img = cx * temp(1) / temp(3);
% % % % % % % % %         y_img = cy * temp(2) / temp(3);
% % % % % % % % % 
% % % % % % % % %         % this calculation down in OpenCV coordinates but centered
% % % % % % % % %         dx(m) = x_img - imagePoints_original(m,1); %this points to the right
% % % % % % % % %         dy(m) = y_img - imagePoints_original(m,2); %this points down
% % % % % % % % %     end
% % % % % % % % %     
% % % % % % % % %     e = sqrt(dx.^2 + dy.^2);
% % % % % % % % %     
% % % % % % % % %     reprojectionError_original(perStation) = e;
% % % % % % % % %     % remove the image observations with huge errors, because likely blunder
% % % % % % % % %     I = find(e > e_threshold);
% % % % % % % % %     outlierIndex(perStation(I)) = true;
% % % % % % % % %     
% % % % % % % % %     error_vector_mean_before(n) = mean(e);
% % % % % % % % %     error_vector_max_before(n) = max(e);
% % % % % % % % %     e(I) = [];
% % % % % % % % %     error_vector_mean_after(n) = mean(e);
% % % % % % % % %     error_vector_max_after(n) = max(e);

%     figure;
%     plot(1:length(e), e,'b')
%     hold on;
%     plot(1:length(e), e_threshold.*ones(1,length(e)),'r')
%     hold off;
%     e = sqrt(error / (2*length(worldPoints_original(:,1))));
%     error_vector = [error_vector; e];

%     totalInlierError = sqrt(dx(mostInliers).^2 + dy(mostInliers).^2);
%     figure
% %     label(num2cell(
%     q1 = quiver(imagePoints_original(:,1), -imagePoints_original(:,2), dx, -dy, 0,'k');
%     q1.LineWidth = 1;
%     q1.AutoScale ='off';
%     hold on
%     q = quiver(imagePoints_original(I,1), -imagePoints_original(I,2), dx(I), -dy(I), 'r');
%     q.LineWidth = 1;
%     q.AutoScale ='off';
%     hold off
%     legend('Inliers', 'Outliers','Location', 'bestoutside')
%     disp('done')

% % % % % %     
% % % % % %     imagePointsID(mostInliers)
% % % % % %     pho_output = [pho_output; [imagePointsID(mostInliers), ] ];
    
%     error_x = 0;
%     error_y = 0;
%     dx = [];
%     dy = [];
%     for m = 1:length(worldPoints_original(:,1))
%         temp = worldOrientation1 * (worldPoints_original(m,:)' - worldLocation1');
%         x_img = xp -c_x * temp(1) / temp(3);
%         y_img = yp -c_y * temp(2) / temp(3);
%         error_x = error_x + (x_img - imagePoints_original(m,1))^2;
%         error_y = error_y + (y_img + imagePoints_original(m,2))^2;
%         
%         dx = [dx, x_img - imagePoints_original(m,1)];
%         dy = [dy, y_img + imagePoints_original(m,2)];
%     end
%     error = error_x + error_y;
%     e = sqrt(error / (2*length(worldPoints_original(:,1))));
%     error_vector = [error_vector; e];
%     
%     figure
%     quiver(imagePoints_original(:,1), imagePoints_original(:,2), dx', dy')
%        
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

% % % % % % % %     omega = pi;
% % % % % % % %     phi   = 0;
% % % % % % % %     kappa = 0;
% % % % % % % % 
% % % % % % % %     m_11 = cos(phi) * cos(kappa) ;
% % % % % % % %     m_12 = sin(omega) * sin(phi) * cos(kappa) + cos(omega) * sin(kappa) ;
% % % % % % % %     m_13 = -cos(omega) * sin(phi) * cos(kappa) + sin(omega) * sin(kappa) ;
% % % % % % % %     m_21 = -cos(phi) * sin(kappa) ;
% % % % % % % %     m_22 = -sin(omega) * sin(phi) * sin(kappa) + cos(omega) * cos(kappa) ;
% % % % % % % %     m_23 = cos(omega) * sin(phi) * sin(kappa) + sin(omega) * cos(kappa) ;
% % % % % % % %     m_31 = sin(phi) ;
% % % % % % % %     m_32 = -sin(omega) * cos(phi) ;
% % % % % % % %     m_33 = cos(omega) * cos(phi) ;
% % % % % % % % 
% % % % % % % %     M=[m_11 m_12 m_13;
% % % % % % % %        m_21 m_22 m_23;
% % % % % % % %        m_31 m_32 m_33];
% % % % % % % % 
% % % % % % % %     R = M * worldOrientation;
% % % % % % % %     T = worldLocation;
% % % % % % % %     
% % % % % % % %     w = atan2(-R(3,2),R(3,3));
% % % % % % % %     p = asin(R(3,1));
% % % % % % % %     k = atan2(-R(2,1),R(1,1));

    w = x(1);
    p = x(2);
    k = x(3);
    T = [x(4); x(5); x(6)];
    
    w_vector = [w_vector; w];
    p_vector = [p_vector; p];
    k_vector = [k_vector; k];
    
    Xo_vector = [Xo_vector; T(1)];
    Yo_vector = [Yo_vector; T(2)];
    Zo_vector = [Zo_vector; T(3)];
    
    
%     x0 = [w;p;k;T(1);T(2);T(3)];
%     lx = imagePoints_original(:,1);
%     ly = imagePoints_original(:,2);
%     X = worldPoints_original(:,1);
%     Y = worldPoints_original(:,2);
%     Z = worldPoints_original(:,3);
%     f(x0, cx, cy, X, Y, Z, lx, ly);
%     options = optimoptions(@lsqnonlin,'Display', 'off');
%     [x,resnorm,residuals,exitflag,output] = lsqnonlin(@(param) f(param, cx, cy, X, Y, Z, lx, ly), x0, [], [], options);
%     apostStdDev = sqrt(resnorm / (2*length(lx)));
%     
%     figure;
%     plot(imagePointsID, residuals(1:length(lx)),'b.')
%     hold on;
%     plot(imagePointsID, residuals(length(lx)+1:2*length(ly)),'r.')
%     hold off;
    
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

% cx
% nanmean(error_vector)
% nanmedian(error_vector)

hold off
axis equal
xlabel('X')
ylabel('Y')
zlabel('Z')


% plot the reprojection error
% % % % % % % rng(0);
% % % % % % % figure;
% % % % % % % for n = 1:1:length(ID_EOP_unique)
% % % % % % %     I = find(ID_EOP == ID_EOP_unique(n)); % I gives you all the image measurements from that station
% % % % % % % 
% % % % % % %     plot (ID_img(I), reprojectionError_original(I),'.','color',rand(1,3))
% % % % % % %     hold on;
% % % % % % % end
% % % % % % % hold off;
% % % % % % % title('Reprojection Errors Before RANSAC')
% % % % % % % xlabel('Point ID')
% % % % % % % ylabel('Errors [pix]')

rng(0);
figure;
for n = 1:1:length(ID_EOP_unique)
    I = find(ID_EOP == ID_EOP_unique(n)); % I, gives you all the image measurements from that station

    plot (ID_img(I), reprojectionError_original2(I,1),'.','color',rand(1,3))
    hold on;
end
hold off;
title('Reprojection x Errors Before Outlier Removal')
xlabel('Point ID')
ylabel('Errors [pix]')

rng(0);
figure;
for n = 1:1:length(ID_EOP_unique)
    I = find(ID_EOP == ID_EOP_unique(n)); % I gives you all the image measurements from that station

    plot (ID_img(I), reprojectionError_original2(I,2),'.','color',rand(1,3))
    hold on;
end
hold off;
title('Reprojection y Errors Before Outlier Removal')
xlabel('Point ID')
ylabel('Errors [pix]')

rng(0);
figure;
for n = 1:1:length(ID_EOP_unique)
    I = find(ID_EOP == ID_EOP_unique(n)); % I, gives you all the image measurements from that station

    plot (sqrt(img(I,1).^2+img(I,2).^2), reprojectionError_original2(I,1),'.','color',rand(1,3))
    hold on;
end
hold off;
title('Reprojection x Errors Before Outlier Removal')
xlabel('Radial Distance')
ylabel('Errors [pix]')

rng(0);
figure;
for n = 1:1:length(ID_EOP_unique)
    I = find(ID_EOP == ID_EOP_unique(n)); % I gives you all the image measurements from that station

    plot (sqrt(img(I,1).^2+img(I,2).^2), reprojectionError_original2(I,2),'.','color',rand(1,3))
    hold on;
end
hold off;
title('Reprojection y Errors Before Outlier Removal')
xlabel('Radial Distance')
ylabel('Errors [pix]')


% now plot it with the colors corresponding to the point
rng(0);
figure;
subplot(1,2,1)
ID_img_unique = unique(ID_img);
spread = zeros(length(ID_img_unique),1);
for n = 1:1:length(ID_img_unique)
    I = find(ID_img == ID_img_unique(n)); % I, gives you all the image measurements from that station

    spread(n) = std(reprojectionError_original2(I,1));
    
    plot (ID_img(I), reprojectionError_original2(I,1),'.','color',rand(1,3))
    hold on;
end
hold off;
ylim([-max(abs(reprojectionError_original2(:,1))), max(abs(reprojectionError_original2(:,1)))]);
title('Reprojection x Errors Before Outlier Removal')
xlabel('Point ID')
ylabel('Errors [pix]')

subplot(1,2,2)
plot(ID_img_unique, spread)
hold on;
plot(ID_img_unique, ones(length(ID_img_unique)) * (mean(spread)+ 3.091*std(spread)), 'r')
hold off;
legend('Spread of the residuals for each target point', '99.9% confidence level')
xlabel('Target ID')
ylabel('Spread of x residuals')

% now plot it with the colors corresponding to the point
rng(0);
figure;
subplot(1,2,1)
ID_img_unique = unique(ID_img);
spread = zeros(length(ID_img_unique),1);
for n = 1:1:length(ID_img_unique)
    I = find(ID_img == ID_img_unique(n)); % I, gives you all the image measurements from that station

    spread(n) = std(reprojectionError_original2(I,2));
    
    plot (ID_img(I), reprojectionError_original2(I,2),'.','color',rand(1,3))
    hold on;
end
hold off;
ylim([-max(abs(reprojectionError_original2(:,2))), max(abs(reprojectionError_original2(:,2)))]);
title('Reprojection y Errors Before Outlier Removal')
xlabel('Point ID')
ylabel('Errors [pix]')

subplot(1,2,2)
plot(ID_img_unique, spread)
hold on;
plot(ID_img_unique, ones(length(ID_img_unique)) * (mean(spread)+ 3.091*std(spread)), 'r')
hold off;
legend('Spread of the residuals for each target point', '99.9% confidence level', 'Location', 'bestoutside')
xlabel('Target ID')
ylabel('Spread of y residuals')

% Remove outliers from each individual photo
% do a global residual removal using residuals in x and y directly
threshold_x = mean(reprojectionError_original2(:,1)) + 3.091*std(reprojectionError_original2(:,1));
[sig,mu,mah,outliers,s]  = robustcov(reprojectionError_original2(:,1));
robustThreshold_x = mu + 3.091*sqrt(sig);
I = find(abs(reprojectionError_original2(:,1)) > robustThreshold_x);

% I = find(reprojectionError_original2(:,1) > threshold_x);

threshold_y = mean(reprojectionError_original2(:,2)) + 3.091*std(reprojectionError_original2(:,2));
% J = find(reprojectionError_original2(:,2) > threshold_y);
[sig,mu,mah,outliers,s]  = robustcov(reprojectionError_original2(:,2));
robustThreshold_y = mu + 3.091*sqrt(sig);
J = find(abs(reprojectionError_original2(:,2)) > robustThreshold_y);

outlierIndex = unique([I;J]); % final, most important index for removing outliers


% % % % % % Remove outliers from each individual photo
    ID_img(outlierIndex) = [];
    ID_EOP(outlierIndex) = [];
    img(outlierIndex,:) = [];
    img_stdDev(outlierIndex,:) = []; 
    img_corr(outlierIndex,:) = [];
    reprojectionError_original(outlierIndex,:) = [];
    reprojectionError_original2(outlierIndex,:) = [];
    
rng(0);
figure;
for n = 1:1:length(ID_EOP_unique)
    I = find(ID_EOP == ID_EOP_unique(n)); % I gives you all the image measurements from that station

    plot (ID_img(I), reprojectionError_original2(I,1),'.','color',rand(1,3))
    hold on;
end
hold off;
title('Reprojection x Errors After Outlier Removal')
xlabel('Point ID')
ylabel('Errors [pix]')

rng(0);
figure;
for n = 1:1:length(ID_EOP_unique)
    I = find(ID_EOP == ID_EOP_unique(n)); % I gives you all the image measurements from that station

    plot (ID_img(I), reprojectionError_original2(I,2),'.','color',rand(1,3))
    hold on;
end
hold off;
title('Reprojection y Errors After Outlier Removal')
xlabel('Point ID')
ylabel('Errors [pix]')


% figure;
% plot(1:length(error_vector_mean_before), error_vector_mean_before,'b')
% hold on
% plot(1:length(error_vector_max_before), error_vector_max_before,'r')
% plot(1:length(error_vector_mean_after), error_vector_mean_after,'m')
% plot(1:length(error_vector_max_after), error_vector_max_after,'k')
% hold off
% title('Reprojection error plot')
% xlabel('Image #')
% ylabel('e [pix]')
% legend('mean e before', 'max e before', 'mean e after', 'max e after', 'Location', 'bestoutside')

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


% flip the output for y_img because that's the notation my C++ program needs
% Note we are outputting -y instead of +y. The original of x and y is still
% the left upper corner, but the  y is just flipped in the output
photoFile = [double(ID_img), double(ID_EOP), img(:,1)+xp, -(img(:,2)-yp), img_stdDev, img_corr];

disp('Writing screened *.pho file: pointID, frameID, x, y, xStdDev, yStdDev, xCorr, yCorr')
out=fopen([pathname, outputPhoFilename],'w');
for n = 1:length(photoFile(:,1))
    fprintf(out, '%d \t %d \t %f \t %f \t %f \t %f \t %f \t %f\n', photoFile(n,1),photoFile(n,2),photoFile(n,3),photoFile(n,4),photoFile(n,5),photoFile(n,6),photoFile(n,7),photoFile(n,8));
end
fclose(out);

OPKXYZ = [EOPID_vector, ones(length(EOPID_vector),1)*1, Xo_vector, Yo_vector, Zo_vector, w_vector, p_vector, k_vector];

disp('Writing to *.eop file Xo, Yo, Zo (mm), w, p, k(deg)')
out=fopen([pathname, outputEOPFilename],'w');
for n = 1:length(OPKXYZ(:,1))
    fprintf(out, '%d \t %d \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', OPKXYZ(n,1),OPKXYZ(n,2),OPKXYZ(n,3),OPKXYZ(n,4),OPKXYZ(n,5),OPKXYZ(n,6)*180/pi,OPKXYZ(n,7)*180/pi,OPKXYZ(n,8)*180/pi);
end
fclose(out);

XYZOut = [worldPointsID, worldPoints, XYZ_stdDev];
disp('Writing to *.xyz file X, Y, Z reduced to its centroid')
out=fopen([pathname, outputXYZFilename],'w');
for n = 1:length(XYZOut(:,1))
    fprintf(out, '%d \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', XYZOut(n,1),XYZOut(n,2),XYZOut(n,3),XYZOut(n,4),XYZOut(n,5),XYZOut(n,6),XYZOut(n,7));
end
fclose(out);

missing_obj = unique(missing_obj)

disp("Success ^-^")

% unknowns params u x 1
function [v] = collinearityResection(param, cx, cy, X, Y, Z, lx, ly)

    omega = param(1);
    phi   = param(2);
    kappa = param(3);
    Xo    = param(4);
    Yo    = param(5);
    Zo    = param(6);

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

    T = [param(4); param(5); param(6)];
    
    fx = zeros(length(X),1);
    fy = zeros(length(X),1);
    for m=1:length(X)
        XYZ = [X(m); Y(m); Z(m)];
        temp = M * (XYZ - T);
        %temp = M * (XYZ - T);
        x_img = -cx * temp(1) / temp(3);
        y_img = -cy * temp(2) / temp(3);

        fx(m) = x_img - lx(m);
        fy(m) = y_img + ly(m);
%         dx = [dx, x_img - imagePoints_original(m,1)];
%         dy = [dy, y_img + imagePoints_original(m,2)];   
    end
    v = [fx; fy;];
end

function [v] = SinglePhotoCalibration(param, X, Y, Z, lx, ly)

    omega = param(1);
    phi   = param(2);
    kappa = param(3);
    Xo    = param(4);
    Yo    = param(5);
    Zo    = param(6);
    c     = param(7);

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

    T = [param(4); param(5); param(6)];
    
    
    fx = zeros(length(X),1);
    fy = zeros(length(X),1);
    for m=1:length(X)
        XYZ = [X(m); Y(m); Z(m)];
        temp = M * (XYZ - T);
        %temp = M * (XYZ - T);
        x_img = -c * temp(1) / temp(3);
        y_img = -c * temp(2) / temp(3);

        fx(m) = x_img - lx(m);
        fy(m) = y_img + ly(m);
%         dx = [dx, x_img - imagePoints_original(m,1)];
%         dy = [dy, y_img + imagePoints_original(m,2)];   
    end
    v = [fx; fy;];
end



function [v] = ResectionPhotoCalibration(param, X, Y, Z, lx, ly)

    omega = param(1);
    phi   = param(2);
    kappa = param(3);
    Xo    = param(4);
    Yo    = param(5);
    Zo    = param(6);
    xp    = param(7);
    yp    = param(8);
    c     = param(9);

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

    T = [param(4); param(5); param(6)];
    
    
    fx = zeros(length(X),1);
    fy = zeros(length(X),1);
    for m=1:length(X)
        XYZ = [X(m); Y(m); Z(m)];
        temp = M * (XYZ - T);
        %temp = M * (XYZ - T);
        x_img = -c * temp(1) / temp(3);
        y_img = -c * temp(2) / temp(3);
        
% % % % %   // collinearity condition
% % % % %   T x = -IOP[2] * XTemp / ZTemp;
% % % % %   T y = -IOP[2] * YTemp / ZTemp;
% % % % % 
% % % % % //   std::cout<<"x, y: "<<x+T(xp_)<<", "<<y+T(yp_)<<std::endl;
% % % % % //   std::cout<<"x_obs, y_obs: "<<T(x_)<<", "<<T(y_)<<std::endl;
% % % % % 
% % % % % 
% % % % %   // camera correction model AP = a1, a2, k1, k2, k3, p1, p2, ...
% % % % %   T x_bar = T(x_) - T(xp_);
% % % % %   T y_bar = T(y_) - T(yp_);
% % % % %   T r = sqrt(x_bar*x_bar + y_bar*y_bar);
% % % % % 
% % % % % //   T delta_x = x_bar*(AP[2]*pow(r,2.0)+AP[3]*pow(r,4.0)+AP[4]*pow(r,6.0)) + AP[5]*(pow(r,2.0)+T(2.0)*pow(x_bar,2.0))+T(2.0)*AP[6]*x_bar*y_bar + AP[0]*x_bar+AP[1]*y_bar;
% % % % % //   T delta_y = y_bar*(AP[2]*pow(r,2.0)+AP[3]*pow(r,4.0)+AP[4]*pow(r,6.0)) + AP[6]*(pow(r,2.0)+T(2.0)*pow(y_bar,2.0))+T(2.0)*AP[5]*x_bar*y_bar;
% % % % % 
% % % % %   T delta_x = x_bar*(AP[2]*r*r+AP[3]*r*r*r*r+AP[4]*r*r*r*r*r*r) + AP[5]*(r*r+T(2.0)*x_bar*x_bar)+T(2.0)*AP[6]*x_bar*y_bar + AP[0]*x_bar+AP[1]*y_bar;
% % % % %   T delta_y = y_bar*(AP[2]*r*r+AP[3]*r*r*r*r+AP[4]*r*r*r*r*r*r) + AP[6]*(r*r+T(2.0)*y_bar*y_bar)+T(2.0)*AP[5]*x_bar*y_bar;
% % % % % 
% % % % % 
% % % % %   T x_true = x + IOP[0] + delta_x;
% % % % %   T y_true = y + IOP[1] + delta_y;


        fx(m) = x_img - lx(m);
        fy(m) = y_img + ly(m);
%         dx = [dx, x_img - imagePoints_original(m,1)];
%         dy = [dy, y_img + imagePoints_original(m,2)];   
    end
    v = [fx; fy;];
end


% unknowns params u x 1
function [v] = stereographicResection(param, cx, cy, X, Y, Z, lx, ly)

    omega  = param(1);
    phi    = param(2);
    kappa  = param(3);
    Xo     = param(4);
    Yo     = param(5);
    Zo     = param(6);
    radius = param(7);

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

    T = [param(4); param(5); param(6)];
    
    fx = zeros(length(X),1);
    fy = zeros(length(X),1);
    for m=1:length(X)
        XYZ = [X(m); Y(m); Z(m)];
        XYZ_s = M * (XYZ - T);

        lambda = radius / sqrt(XYZ_s(1)*XYZ_s(1) + XYZ_s(2)*XYZ_s(2) + XYZ_s(3)*XYZ_s(3));
        
        XYZ_c = lambda * XYZ_s;

        % stereographic projection of point on sphere onto image place
        x_img = (radius + cx)/(radius - XYZ_c(3)) * XYZ_c(1);
        y_img = (radius + cy)/(radius - XYZ_c(3)) * XYZ_c(2);
        
        fx(m) = x_img - lx(m);
        fy(m) = y_img + ly(m);
%         dx = [dx, x_img - imagePoints_original(m,1)];
%         dy = [dy, y_img + imagePoints_original(m,2)];   
    end
    v = [fx; fy;];
end