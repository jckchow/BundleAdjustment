function photogrammetricSpatialResection13
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
% outputMATLABWorkspace = 'nikon.mat'

pathname = 'C:\Users\jckch\OneDrive - University of Calgary\Google Drive\Omni-Directional Cameras\Data\GoPro\jacky_2020_03_31\';
filenameIOP = 'gopro.iop'; % treat as constant
% filenameXYZ = 'goproLowWeight_centred.xyz'; % treat as constant
filenameXYZ = 'gopro.xyz'; % treat as constant
filenameImage = 'gopro.pho'; % outliers will be removed
outputEOPFilename = 'gopro.eop';
outputPhoFilename = 'gopro_screened.pho'; % without outliers based on single photo resection RANSAC
outputXYZFilename = 'gopro_centred.xyz';
outputMATLABWorkspace = 'gopro_stereographic.mat';

% 1 = collinearity, 2 = stereographic projection model
mode = 2;

% run a BIG resection based calibration of IOP and EOP simultaneously using all images at the end
doCalibration = false;

% 1 = KNN, 2 = rNN
filterMode = 1;
KNN = 14; % using the # of img points nearest the principal point, assumed to be least effected by distortions
rFilterDist = 800;

% Single photo resection
% 1 = random initialization, 2 = nChoosek, where numRANSACIter will be set automatically
ransacMode = 2;
numRANSACIter = 1000; % number of RANSAC iterations to do

%% Manually load a sample picture to choose the size of bounding box to accept
I = imread('C:/Users/jckch/OneDrive - University of Calgary/Google Drive/Omni-Directional Cameras/Data/Photos/GOPR0368.JPG');
figure
subplot(2,2,1)
imshow(I)
[rows, columns, numberOfColorChannels] = size(I);
hold on;
lineSpacing = 200; % Whatever you want.
for row = 1 : lineSpacing : rows
    line([1, columns], [row, row], 'Color', 'r', 'LineWidth', 1);
end
for col = 1 : lineSpacing : columns
    line([col, col], [1, rows], 'Color', 'r', 'LineWidth', 1);
end
title(['rDist ~ ', num2str(floor( sqrt((rows/2).^2 + (columns/2).^2) ))])
I = I(floor(rows/2) - floor(rows/4):floor(rows/2) + floor(rows/4), floor(columns/2) - floor(columns/4):floor(columns/2) + floor(columns/4));
subplot(2,2,2)
imshow(I)
[rows, columns, numberOfColorChannels] = size(I);
hold on;
lineSpacing = lineSpacing/2; % Whatever you want.
for row = 1 : lineSpacing : rows
    line([1, columns], [row, row], 'Color', 'r', 'LineWidth', 1);
end
for col = 1 : lineSpacing : columns
    line([col, col], [1, rows], 'Color', 'r', 'LineWidth', 1);
end
title(['rDist ~ ', num2str(floor( sqrt((rows/2).^2 + (columns/2).^2) ))])
I = I(floor(rows/2) - floor(rows/4):floor(rows/2) + floor(rows/4), floor(columns/2) - floor(columns/4):floor(columns/2) + floor(columns/4));
subplot(2,2,3)
imshow(I)
[rows, columns, numberOfColorChannels] = size(I);
hold on;
lineSpacing = lineSpacing/2; % Whatever you want.
for row = 1 : lineSpacing : rows
    line([1, columns], [row, row], 'Color', 'r', 'LineWidth', 1);
end
for col = 1 : lineSpacing : columns
    line([col, col], [1, rows], 'Color', 'r', 'LineWidth', 1);
end
title(['rDist ~ ', num2str(floor( sqrt((rows/2).^2 + (columns/2).^2) ))])
I = I(floor(rows/2) - floor(rows/4):floor(rows/2) + floor(rows/4), floor(columns/2) - floor(columns/4):floor(columns/2) + floor(columns/4));
subplot(2,2,4)
imshow(I)
[rows, columns, numberOfColorChannels] = size(I);
hold on;
lineSpacing = lineSpacing/2; % Whatever you want.
for row = 1 : lineSpacing : rows
    line([1, columns], [row, row], 'Color', 'r', 'LineWidth', 1);
end
for col = 1 : lineSpacing : columns
    line([col, col], [1, rows], 'Color', 'r', 'LineWidth', 1);
end
title(['rDist ~ ', num2str(floor( sqrt((rows/2).^2 + (columns/2).^2) ))])
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

radiusInitial = (cx+cy)/2; % if mode is 2

% read in the object space file
in=fopen([pathname, filenameXYZ],'r');
data=textscan(in,'%d %f %f %f %f %f %f','headerlines',0);
fclose(in);

ID_XYZ = data{1};
XYZ = [data{2}, data{3} data{4}];
XYZ_stdDev = [data{5}, data{6} data{7}];
XYZ(:,1) = XYZ(:,1) - mean(XYZ(:,1)); % reduce to centroid
XYZ(:,2) = XYZ(:,2) - mean(XYZ(:,2));
XYZ(:,3) = XYZ(:,3) - mean(XYZ(:,3));


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

% determine the min/max targets in images
minTargetsInImage = 1E10;
maxTargetsInImage = 0;
for n = 1:1:length(ID_EOP_unique)
    I = find(ID_EOP == ID_EOP_unique(n)); % I gives you all the image measurements from that station
    if (length(I) < minTargetsInImage)
        minTargetsInImage = length(I);
    end
    if (length(I) > maxTargetsInImage)
        maxTargetsInImage = length(I);
    end
end
disp('Done reading in all files')
disp(['   Min targets, hence min K: ', num2str(minTargetsInImage)])
disp(['   Max targets: ', num2str(maxTargetsInImage)])

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

allUnknowns = [xp0; yp0; (cx+cy)/2];
dataForCalibration = [];
dataForTesting = [];
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
    ID_img_original = imagePointsID;
    ID_EOP_original = ones(length(imagePointsID),1) .* double(ID_EOP_unique(n));
    imagePoints_original = imagePoints; % all image points even not filtered
    worldPoints_original = worldPoints;
    dist = sqrt(imagePoints(:,1).^2 + (-imagePoints(:,2)).^2);
    %     figure;
    %     plot((imagePoints(:,1)-xp), (-(imagePoints(:,2)+yp)),'b.')
    [dist_sorted,K] = sort(dist);
    %     KK = find(K <= KNN); % use only the K points close to the middle of the image
    if (filterMode == 1)
        KK = K(1:KNN);
        disp('      kNN filter mode')
    end
    
    if (filterMode == 2)
        KK = K(find(dist_sorted < rFilterDist));
        disp('      rNN filter mode')
    end
    
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
    if (status ~= 0)
        disp(['         Warning: I have to increase the RANSAC dist threshold'])
        [worldOrientation,worldLocation,inlierIdx,status] = estimateWorldCameraPose(imagePoints,worldPoints,cameraParams,'MaxNumTrials', 10000, 'MaxReprojectionError', 10);
    end
    
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
        x0 = [w;p;k;T(1);T(2);T(3);radiusInitial]; % initial unknown vector
    end
    
    tic;
    if (mode == 1)
        m = 3; % min number of points
    end
    
    if (mode == 2)
        m = 4; % min number of points
    end
    
    if (ransacMode == 2)
        % only good if k is <=15
        disp(['         nChoosek is: ', num2str(nchoosek(length(imagePoints(:,1)), m)) ]);
        r = nchoosek(1:length(imagePoints(:,1)), m);
        numRANSACIter = length(r(:,1));
        disp(['         Setting numRANSACIter to: ', num2str(numRANSACIter) ]);
    end
    for ransacIter = 1:numRANSACIter
        
        if (ransacMode == 1)
            r = randperm(length(imagePoints(:,1)));
            inliers = r(1:m);
        end
        
        if (ransacMode == 2)
            inliers = r(ransacIter,:);
        end
        
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
            [x,resnorm,residuals,exitflag,output] = lsqnonlin(@(param) stereographicResection(param, X, Y, Z, lx, ly), x0, lb, [], options);
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
        if (ransacMode == 1)
            inliers = r;
        end
        
        if (ransacMode == 2)
            inliers = 1:length(imagePoints(:,1));
        end
        
        lx = imagePoints(inliers,1);
        ly = imagePoints(inliers,2);
        X = worldPoints(inliers,1);
        Y = worldPoints(inliers,2);
        Z = worldPoints(inliers,3);
        
        if (mode == 1)
            v = collinearityResection(x, cx, cy, X, Y, Z, lx, ly);
        end
        
        if (mode == 2)
            v = stereographicResection(x, X, Y, Z, lx, ly);
        end
        
        residuals = reshape(v,length(X),2);
        
        I = find(abs(residuals(:,1))<inlierThreshold);
        J = find(abs(residuals(I,2))<inlierThreshold);
        inliers = I(J);
        
        testError = sqrt(v(inliers)'*v(inliers)) / length(inliers); % this is the residual norm for all
        if(length(inliers) > length(mostInliers))
            mostInliersErrors = testError;
            if (ransacMode == 1)
                mostInliers = r(inliers);
            end
            if (ransacMode == 2)
                mostInliers = inliers;
            end
        elseif length(inliers) == length(mostInliers)
            if (testError < mostInliersErrors)
                mostInliersErrors = testError;
                if (ransacMode == 1)
                    mostInliers = r(inliers);
                end
                if (ransacMode == 2)
                    mostInliers = inliers;
                end
            end
        end
    end
    
    disp(['      Number of RANSAC inliers for estimating EOP: ', num2str(length(mostInliers))])
    
    
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
        lb = [-1E10; -1E10; -1E10; -1E10; -1E10; -1E10; 1E-9];
        [x,resnorm,residuals,exitflag,output] = lsqnonlin(@(param) stereographicResection(param, X, Y, Z, lx, ly), x0, lb, [], options);
        disp(['         C = r+c: ', num2str(x0(7)), ' --> ', num2str(x(7))]);
    end
    
    apostStdDev = sqrt(resnorm / (2*length(lx)));
    %         apostStdDev = residuals * img_stdDev';
    
    disp(['         Reprojection error of inliers: ', num2str(apostStdDev)])
    toc
    
    % only inliers are stored for global resection calibration later on
    dataForCalibration = [dataForCalibration; [n.*ones(length(lx),1), lx, ly, X, Y, Z, reshape(residuals,length(X),2)]];
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
        residuals = stereographicResection(x, X, Y, Z, lx, ly);
    end
    
    residuals = reshape(residuals,length(X),2);
    
    reprojectionError_original2(perStation,:) = residuals;
    
    dataForTesting = [dataForTesting; [n.*ones(length(lx),1), lx, ly, X, Y, Z, reshape(residuals,length(X),2)], double(ID_img_original), ID_EOP_original ];
    
    
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
    
    allUnknowns = [allUnknowns;w;p;k;T];
    
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

save([pathname, outputMATLABWorkspace])

% plot the inliers
figure;
plot(img(:,1),-img(:,2),'r.', 'MarkerSize',10)
hold on;
plot(dataForCalibration(:,2), -dataForCalibration(:,3),'b.');
hold off;
xlim([-xp xp])
ylim([yp -yp])
title('Distribution of inlier points')

%% Big calibration
load("C:\Users\jckch\OneDrive - University of Calgary\Google Drive\Omni-Directional Cameras\Data\GoPro\jacky_2020_03_31\gopro_stereographic.mat")

options = optimoptions(@lsqnonlin,'Display', 'iter', 'MaxIter', 500, 'MaxFunEvals', 1E10);
if (mode == 2)
    if (doCalibration)
        %     v = stereographicResectionCalibration(allUnknowns, dataForCalibration(:,4), dataForCalibration(:,5), dataForCalibration(:,6), dataForCalibration(:,2), dataForCalibration(:,3), dataForCalibration(:,1))
        lb = ones(length(allUnknowns),1) .* -1E10;
        lb(3) = 0.0; % C
        [x,resnorm,residuals,exitflag,output] = lsqnonlin(@(param) stereographicResectionCalibration(param, dataForCalibration(:,4), dataForCalibration(:,5), dataForCalibration(:,6), dataForCalibration(:,2), dataForCalibration(:,3), dataForCalibration(:,1)), allUnknowns, lb, [], options);
%         [x,resnorm,residuals,exitflag,output] = lsqnonlin(@(param) fisheyeEquidistantResectionCalibrationAP(param, dataForCalibration(:,4), dataForCalibration(:,5), dataForCalibration(:,6), dataForCalibration(:,2), dataForCalibration(:,3), dataForCalibration(:,1)), allUnknowns, lb, [], options); 
        
        disp(['Apost Var: ', num2str(resnorm/length(residuals))]);
        disp(['Apost StdDev: ', num2str(sqrt(resnorm/length(residuals)))]);
        v = reshape(residuals,length(residuals)/2,2);
        disp(['Mean   vx: ', num2str(mean(v(:,1)))]);
        disp(['StdDev vx: ', num2str(std(v(:,1)))]);
        disp(['RMSE   vx: ', num2str(sqrt(mean(v(:,1).^2)))]);
        disp(['Mean   vy: ', num2str(mean(v(:,2)))]);
        disp(['StdDev vy: ', num2str(std(v(:,2)))]);
        disp(['RMSE   vy: ', num2str(sqrt(mean(v(:,2).^2)))]);
        disp(['xp: ', num2str(x(1))]);
        disp(['yp: ', num2str(x(2))]);
        disp([' c: ', num2str(x(3))]);
        
        numAP = 3;
        allUnknowns = x;
        allUnknowns = [allUnknowns(1:3); zeros(numAP,1); allUnknowns(4:end)];% IOP, AP, EOP
        lb = ones(length(allUnknowns),1) .* -1E10;
        lb(3) = 1000.0; % C
        [x,resnorm,residuals,exitflag,output] = lsqnonlin(@(param) stereographicResectionCalibrationAP(param, dataForCalibration(:,4), dataForCalibration(:,5), dataForCalibration(:,6), dataForCalibration(:,2), dataForCalibration(:,3), dataForCalibration(:,1)), allUnknowns, lb, [], options);
        disp(['Apost Var: ', num2str(resnorm/length(residuals))]);
        disp(['Apost StdDev: ', num2str(sqrt(resnorm/length(residuals)))]);
        v = reshape(residuals,length(residuals)/2,2);
        disp(['Mean   vx: ', num2str(mean(v(:,1)))]);
        disp(['StdDev vx: ', num2str(std(v(:,1)))]);
        disp(['RMSE   vx: ', num2str(sqrt(mean(v(:,1).^2)))]);
        disp(['Mean   vy: ', num2str(mean(v(:,2)))]);
        disp(['StdDev vy: ', num2str(std(v(:,2)))]);
        disp(['RMSE   vy: ', num2str(sqrt(mean(v(:,2).^2)))]);
        disp(['xp: ', num2str(x(1))]);
        disp(['yp: ', num2str(x(2))]);
        disp([' c: ', num2str(x(3))]);

    end
    v = reshape(residuals,length(residuals)/2,2);
    e = sqrt( v(:,1).^2 + v(:,2).^2 );
    dist = sqrt(dataForCalibration(:,2).^2 + dataForCalibration(:,3).^2);
    alpha = incidenceAngle(x, dataForCalibration(:,4), dataForCalibration(:,5), dataForCalibration(:,6), dataForCalibration(:,2), dataForCalibration(:,3), dataForCalibration(:,1));

    figure;
    subplot(2,3,1)
    plot(v(:,1), v(:,2), '.');
    xlabel('v_x')
    ylabel('v_y')
    title('Only inlier points after calibration')
    subplot(2,3,2)
    plot(dataForCalibration(:,2), v(:,1), '.');
    xlabel('x')
    ylabel('v_x')
    subplot(2,3,3)
    plot(dataForCalibration(:,3), v(:,2), '.');
    xlabel('y')
    ylabel('v_y')
    subplot(2,3,4)
    plot(dist, e, '.');
    xlabel('Radial Distance')
    ylabel('Norm residual')
    subplot(2,3,5)
    plot(dist, v(:,1), '.');
    xlabel('Radial Distance')
    ylabel('v_x')
    subplot(2,3,6)
    plot(dist, v(:,2), '.');
    xlabel('Radial Distance')
    ylabel('v_y')
    
    figure;
    subplot(1,3,1)
    plot(alpha*180/pi, e, '.');
    xlabel('Incidence Angle')
    ylabel('Norm residual')
    title('Only inlier points after calibration')
    subplot(1,3,2)
    plot(alpha*180/pi, v(:,1), '.');
    xlabel('Incidence Angle')
    ylabel('v_x')
    subplot(1,3,3)
    plot(alpha*180/pi, v(:,2), '.');
    xlabel('Incidence Angle')
    ylabel('v_y')
    
    figure;
    pose = x(4:end);
%     pose = x(7:end);
    pose = reshape(pose, 6, length(pose)/6)';
    
    w_vector  = pose(:,1);
    p_vector  = pose(:,2);
    k_vector  = pose(:,3);
    Xo_vector = pose(:,4);
    Yo_vector = pose(:,5);
    Zo_vector = pose(:,6);
    pcshow(XYZ,'VerticalAxis','Y','VerticalAxisDir','down', ...
        'MarkerSize',100);
    hold on;
    for i = 1:length(pose(:,1))
        
        omega = -pi;
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
        
        omega = pose(i,1);
        phi   = pose(i,2);
        kappa = pose(i,3);
        
        m_11 = cos(phi) * cos(kappa) ;
        m_12 = sin(omega) * sin(phi) * cos(kappa) + cos(omega) * sin(kappa) ;
        m_13 = -cos(omega) * sin(phi) * cos(kappa) + sin(omega) * sin(kappa) ;
        m_21 = -cos(phi) * sin(kappa) ;
        m_22 = -sin(omega) * sin(phi) * sin(kappa) + cos(omega) * cos(kappa) ;
        m_23 = cos(omega) * sin(phi) * sin(kappa) + sin(omega) * cos(kappa) ;
        m_31 = sin(phi) ;
        m_32 = -sin(omega) * cos(phi) ;
        m_33 = cos(omega) * cos(phi) ;
        
        R=[m_11 m_12 m_13;
            m_21 m_22 m_23;
            m_31 m_32 m_33];
        
        worldOrientation = M * R;
        worldLocation = [pose(i,4), pose(i,5), pose(i,6)];
        
        plotCamera('Size',100,'Orientation',worldOrientation,'Location',...
            worldLocation);
    end
    hold off;
    xlabel('X')
    ylabel('Y')
    zlabel('Z')
    title('Post-Calibration Pose')
    axis equal
    
    % now plot all the data for seeing the residuals, this includes
    % non-inliers used during calibration
    alpha = incidenceAngle(x, dataForTesting(:,4), dataForTesting(:,5), dataForTesting(:,6), dataForTesting(:,2), dataForTesting(:,3), dataForTesting(:,1));
%     residuals = stereographicResectionCalibrationAP(x, dataForTesting(:,4), dataForTesting(:,5), dataForTesting(:,6), dataForTesting(:,2), dataForTesting(:,3), dataForTesting(:,1));
%     residuals = stereographicResectionCalibration(x, dataForTesting(:,4), dataForTesting(:,5), dataForTesting(:,6), dataForTesting(:,2), dataForTesting(:,3), dataForTesting(:,1));
    residuals = fisheyeEquidistantResectionCalibrationAP(x, dataForTesting(:,4), dataForTesting(:,5), dataForTesting(:,6), dataForTesting(:,2), dataForTesting(:,3), dataForTesting(:,1));

    v = reshape(residuals,length(residuals)/2,2);
    e = sqrt( v(:,1).^2 + v(:,2).^2 );
    dist = sqrt(dataForTesting(:,2).^2 + dataForTesting(:,3).^2);
    disp(['Apost Var: ', num2str(norm(residuals)/length(residuals))]);
    disp(['Apost StdDev: ', num2str(sqrt(norm(residuals)/length(residuals)))]);
    v = reshape(residuals,length(residuals)/2,2);
    disp(['Mean   vx: ', num2str(mean(v(:,1)))]);
    disp(['StdDev vx: ', num2str(std(v(:,1)))]);
    disp(['RMSE   vx: ', num2str(sqrt(mean(v(:,1).^2)))]);
    disp(['Mean   vy: ', num2str(mean(v(:,2)))]);
    disp(['StdDev vy: ', num2str(std(v(:,2)))]);
    disp(['RMSE   vy: ', num2str(sqrt(mean(v(:,2).^2)))]);   
    
    figure;
    subplot(2,3,1)
    plot(v(:,1), v(:,2), '.');
    xlabel('v_x')
    ylabel('v_y')
    title('All points after calibration')
    subplot(2,3,2)
    plot(dataForTesting(:,2), v(:,1), '.');
    xlabel('x')
    ylabel('v_x')
    subplot(2,3,3)
    plot(dataForTesting(:,3), v(:,2), '.');
    xlabel('y')
    ylabel('v_y')
    subplot(2,3,4)
    plot(dist, e, '.');
    xlabel('Radial Distance')
    ylabel('Norm residual')
    subplot(2,3,5)
    plot(dist, v(:,1), '.');
    xlabel('Radial Distance')
    ylabel('v_x')
    subplot(2,3,6)
    plot(dist, v(:,2), '.');
    xlabel('Radial Distance')
    ylabel('v_y')
    
    figure;
    subplot(1,3,1)
    plot(alpha*180/pi, e, '.');
    xlabel('Incidence Angle')
    ylabel('Norm residual')
    title('All points after calibration')
    subplot(1,3,2)
    plot(alpha*180/pi, v(:,1), '.');
    xlabel('Incidence Angle')
    ylabel('v_x')
    subplot(1,3,3)
    plot(alpha*180/pi, v(:,2), '.');
    xlabel('Incidence Angle')
    ylabel('v_y')
    
%     I = find(e<20); % define the threshold for outliers manually
    J = find(abs(v(:,1))<300); % define the threshold for outliers manually
    K = find(abs(v(J,2))<300);
    I = J(K);

    disp(['Total number of points: ', num2str(length(v(:,1)))]);
    disp(['   Number of inliers: ', num2str(length(I))]);
    disp(['   Percent of inliers: ', num2str(100*length(I) / length(v(:,1)))]);
    disp(['Mean   vx: ', num2str(mean(v(I,1)))]);
    disp(['StdDev vx: ', num2str(std(v(I,1)))]);
    disp(['RMSE   vx: ', num2str(sqrt(mean(v(I,1).^2)))]);
    disp(['Mean   vy: ', num2str(mean(v(I,2)))]);
    disp(['StdDev vy: ', num2str(std(v(I,2)))]);
    disp(['RMSE   vy: ', num2str(sqrt(mean(v(I,2).^2)))]);       
    
    figure;
    plot(dataForTesting(:,2),-dataForTesting(:,3),'r.')
    hold on;
    plot(dataForTesting(I,2),-dataForTesting(I,3),'g.')
    hold off;
    legend('Outliers', 'Inliers')
    
    figure;
    subplot(2,3,1)
    plot(v(:,1), v(:,2), 'r.');
    hold on;
    plot(v(I,1), v(I,2), '.');
    hold off
    xlabel('v_x')
    ylabel('v_y')
    title('All points after calibration')
    subplot(2,3,2)
    plot(dataForTesting(:,2), v(:,1), 'r.');
    hold on;
    plot(dataForTesting(I,2), v(I,1), '.');
    hold off
    xlabel('x')
    ylabel('v_x')
    subplot(2,3,3)
    plot(dataForTesting(:,3), v(:,2), 'r.');
    hold on;
    plot(dataForTesting(I,3), v(I,2), '.');    
    hold off;
    xlabel('y')
    ylabel('v_y')
    subplot(2,3,4)
    plot(dist, e, 'r.');
    hold on;
    plot(dist(I), e(I), '.');    
    hold off;
    xlabel('Radial Distance')
    ylabel('Norm residual')
    subplot(2,3,5)
    plot(dist, v(:,1), 'r.');
    hold on;
    plot(dist(I), v(I,1), '.');
    hold off;
    xlabel('Radial Distance')
    ylabel('v_x')
    subplot(2,3,6)
    plot(dist, v(:,2), 'r.');
    hold on;
    plot(dist(I), v(I,2), '.');     
    hold off;
    xlabel('Radial Distance')
    ylabel('v_y')
    
    figure;
    subplot(1,3,1)
    plot(alpha*180/pi, e, 'r.');
    hold on;
    plot(alpha(I)*180/pi, e(I), '.');
    hold off;
    xlabel('Incidence Angle')
    ylabel('Norm residual')
    title('All points after calibration')
    subplot(1,3,2)
    plot(alpha*180/pi, v(:,1), 'r.');
    hold on;
    plot(alpha(I)*180/pi, v(I,1), '.');
    hold off;
    xlabel('Incidence Angle')
    ylabel('v_x')
    subplot(1,3,3)
    plot(alpha*180/pi, v(:,2), 'r.');
    hold on;
    plot(alpha(I)*180/pi, v(I,2), '.');    
    hold off;
    xlabel('Incidence Angle')
    ylabel('v_y')
    
    
    dataForTesting = dataForTesting(I,:);
    v = v(I,:);
    dist = dist(I,:);
    alpha = alpha(I,:);
    e = e(I,:);
% % % % % % %     
% % % % % % %     % All residuals before calibration
% % % % % % %     v = [dataForTesting(:,7), dataForTesting(:,8)];
% % % % % % %     e = sqrt( v(:,1).^2 + v(:,2).^2 );
% % % % % % %     dist = sqrt(dataForTesting(:,2).^2 + dataForTesting(:,3).^2);
% % % % % % %     
% % % % % % %     figure;
% % % % % % %     subplot(2,3,1)
% % % % % % %     plot(v(:,1), v(:,2), '.');
% % % % % % %     xlabel('v_x')
% % % % % % %     ylabel('v_y')
% % % % % % %     title('All points before calibration')
% % % % % % %     subplot(2,3,2)
% % % % % % %     plot(dataForTesting(:,2), v(:,1), '.');
% % % % % % %     xlabel('x')
% % % % % % %     ylabel('v_x')
% % % % % % %     subplot(2,3,3)
% % % % % % %     plot(dataForTesting(:,3), v(:,2), '.');
% % % % % % %     xlabel('y')
% % % % % % %     ylabel('v_y')
% % % % % % %     subplot(2,3,4)
% % % % % % %     plot(dist, e, '.');
% % % % % % %     xlabel('Radial Distance')
% % % % % % %     ylabel('Norm residual')
% % % % % % %     subplot(2,3,5)
% % % % % % %     plot(dist, v(:,1), '.');
% % % % % % %     xlabel('Radial Distance')
% % % % % % %     ylabel('v_x')
% % % % % % %     subplot(2,3,6)
% % % % % % %     plot(dist, v(:,2), '.');
% % % % % % %     xlabel('Radial Distance')
% % % % % % %     ylabel('v_y')
% % % % % % %     
% % % % % % %     
% % % % % % %     
% % % % % % %     % Inliers before calibration
% % % % % % %     v = [dataForCalibration(:,7), dataForCalibration(:,8)];
% % % % % % %     e = sqrt( v(:,1).^2 + v(:,2).^2 );
% % % % % % %     dist = sqrt(dataForCalibration(:,2).^2 + dataForCalibration(:,3).^2);
% % % % % % %     
% % % % % % %     figure;
% % % % % % %     subplot(2,3,1)
% % % % % % %     plot(v(:,1), v(:,2), '.');
% % % % % % %     xlabel('v_x')
% % % % % % %     ylabel('v_y')
% % % % % % %     title('Inliers before calibration')
% % % % % % %     subplot(2,3,2)
% % % % % % %     plot(dataForCalibration(:,2), v(:,1), '.');
% % % % % % %     xlabel('x')
% % % % % % %     ylabel('v_x')
% % % % % % %     subplot(2,3,3)
% % % % % % %     plot(dataForCalibration(:,3), v(:,2), '.');
% % % % % % %     xlabel('y')
% % % % % % %     ylabel('v_y')
% % % % % % %     subplot(2,3,4)
% % % % % % %     plot(dist, e, '.');
% % % % % % %     xlabel('Radial Distance')
% % % % % % %     ylabel('Norm residual')
% % % % % % %     subplot(2,3,5)
% % % % % % %     plot(dist, v(:,1), '.');
% % % % % % %     xlabel('Radial Distance')
% % % % % % %     ylabel('v_x')
% % % % % % %     subplot(2,3,6)
% % % % % % %     plot(dist, v(:,2), '.');
% % % % % % %     xlabel('Radial Distance')
% % % % % % %     ylabel('v_y')
end


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

ID_EOP = dataForTesting(:,10);
ID_img = dataForTesting(:,9);
reprojectionError_original2 = v(:,1:2);

rng(0);
figure;
subplot(1,2,1)
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
subplot(1,2,2);
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
subplot(1,2,1);
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
subplot(1,2,2);
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

% % % % % % Remove outliers from each individual photo
% % % % % % do a global residual removal using residuals in x and y directly
% % % % % threshold_x = mean(reprojectionError_original2(:,1)) + 3.091*std(reprojectionError_original2(:,1));
% % % % % [sig,mu,mah,outliers,s]  = robustcov(reprojectionError_original2(:,1));
% % % % % robustThreshold_x = mu + 3.091*sqrt(sig);
% % % % % I = find(abs(reprojectionError_original2(:,1)) > robustThreshold_x);
% % % % % 
% % % % % % I = find(reprojectionError_original2(:,1) > threshold_x);
% % % % % 
% % % % % threshold_y = mean(reprojectionError_original2(:,2)) + 3.091*std(reprojectionError_original2(:,2));
% % % % % % J = find(reprojectionError_original2(:,2) > threshold_y);
% % % % % [sig,mu,mah,outliers,s]  = robustcov(reprojectionError_original2(:,2));
% % % % % robustThreshold_y = mu + 3.091*sqrt(sig);
% % % % % J = find(abs(reprojectionError_original2(:,2)) > robustThreshold_y);
% % % % % 
% % % % % outlierIndex = unique([I;J]); % final, most important index for removing outliers


% % % % % % % % % % % % Remove outliers from each individual photo
% % % % % % ID_img(outlierIndex) = [];
% % % % % % ID_EOP(outlierIndex) = [];
% % % % % % img(outlierIndex,:) = [];
% % % % % % img_stdDev(outlierIndex,:) = [];
% % % % % % img_corr(outlierIndex,:) = [];
% % % % % % reprojectionError_original(outlierIndex,:) = [];
% % % % % % reprojectionError_original2(outlierIndex,:) = [];
% % % % % % 
% % % % % % rng(0);
% % % % % % figure;
% % % % % % for n = 1:1:length(ID_EOP_unique)
% % % % % %     I = find(ID_EOP == ID_EOP_unique(n)); % I gives you all the image measurements from that station
% % % % % %     
% % % % % %     plot (ID_img(I), reprojectionError_original2(I,1),'.','color',rand(1,3))
% % % % % %     hold on;
% % % % % % end
% % % % % % hold off;
% % % % % % title('Reprojection x Errors After Outlier Removal')
% % % % % % xlabel('Point ID')
% % % % % % ylabel('Errors [pix]')
% % % % % % 
% % % % % % rng(0);
% % % % % % figure;
% % % % % % for n = 1:1:length(ID_EOP_unique)
% % % % % %     I = find(ID_EOP == ID_EOP_unique(n)); % I gives you all the image measurements from that station
% % % % % %     
% % % % % %     plot (ID_img(I), reprojectionError_original2(I,2),'.','color',rand(1,3))
% % % % % %     hold on;
% % % % % % end
% % % % % % hold off;
% % % % % % title('Reprojection y Errors After Outlier Removal')
% % % % % % xlabel('Point ID')
% % % % % % ylabel('Errors [pix]')
% % % % % % 
% % % % % % 
% % % % % % % figure;
% % % % % % % plot(1:length(error_vector_mean_before), error_vector_mean_before,'b')
% % % % % % % hold on
% % % % % % % plot(1:length(error_vector_max_before), error_vector_max_before,'r')
% % % % % % % plot(1:length(error_vector_mean_after), error_vector_mean_after,'m')
% % % % % % % plot(1:length(error_vector_max_after), error_vector_max_after,'k')
% % % % % % % hold off
% % % % % % % title('Reprojection error plot')
% % % % % % % xlabel('Image #')
% % % % % % % ylabel('e [pix]')
% % % % % % % legend('mean e before', 'max e before', 'mean e after', 'max e after', 'Location', 'bestoutside')
% % % % % % 
% % % % % % % disp('Recovered R')
% % % % % % % R
% % % % % % 
% % % % % % % disp('Difference in R')
% % % % % % % R_diff = R_true' * R
% % % % % % 
% % % % % % % w = atan2(-R_diff(3,2),R_diff(3,3));
% % % % % % % p = asin(R_diff(3,1));
% % % % % % % k = atan2(-R_diff(2,1),R_diff(1,1));
% % % % % % 
% % % % % % % disp('Difference in OPK')
% % % % % % % [w,p,k]
% % % % % % 
% % % % % % % Xo_vector = round(Xo_vector,1);
% % % % % % % Yo_vector = round(Yo_vector,1);
% % % % % % % Zo_vector = round(Zo_vector,1);
% % % % % % %
% % % % % % % w_vector = round(w_vector,2);
% % % % % % % p_vector = round(p_vector,2);
% % % % % % % k_vector = round(k_vector,2);
% % % % % % 
% % % % % % 
% % % % % % % OPK = [mode(w_vector),mode(p_vector),mode(k_vector)];
% % % % % % % disp('Recovered T')
% % % % % % % T = [mode(Xo_vector), mode(Yo_vector), mode(Zo_vector)]


% flip the output for y_img because that's the notation my C++ program needs
% Note we are outputting -y instead of +y. The original of x and y is still
% the left upper corner, but the  y is just flipped in the output
% photoFile = [double(ID_img), double(ID_EOP), img(:,1)+xp, -(img(:,2)-yp), img_stdDev, img_corr];
photoFile = [double(dataForTesting(:,9)), double(dataForTesting(:,10)), dataForTesting(:,2)+xp, -(dataForTesting(:,3)-yp), img_stdDev(1,1:2).*ones(length(dataForTesting(:,3)),2), img_corr(1,1:2).*ones(length(dataForTesting(:,3)),2)];

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

XYZOut = [double(ID_XYZ), XYZ, XYZ_stdDev];
disp('Writing to *.xyz file X, Y, Z reduced to its centroid')
out=fopen([pathname, outputXYZFilename],'w');
for n = 1:length(XYZOut(:,1))
    fprintf(out, '%d \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', XYZOut(n,1),XYZOut(n,2),XYZOut(n,3),XYZOut(n,4),XYZOut(n,5),XYZOut(n,6),XYZOut(n,7));
end
fclose(out);


missing_obj = unique(missing_obj)

disp("Success ^-^")



end

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

T = [Xo; Yo; Zo];

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

function [v] = collinearityResectionPrincipalDistance(param, X, Y, Z, lx, ly)

    omega = param(1);
    phi   = param(2);
    kappa = param(3);
    Xo    = param(4);
    Yo    = param(5);
    Zo    = param(6);
    c     = param(7); % estimates the focal length

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

    T = [Xo; Yo; Zo];

    fx = zeros(length(X),1);
    fy = zeros(length(X),1);
    for m=1:length(X)
        XYZ = [X(m); Y(m); Z(m)];
        temp = M * (XYZ - T);

        x_img = -c * temp(1) / temp(3);
        y_img = -c * temp(2) / temp(3);

        fx(m) = x_img - lx(m);
        fy(m) = y_img + ly(m);
    end
    v = [fx; fy;];
end

function [v] = collinearityResectionCalibration(param, X, Y, Z, lx, ly)


fxVec = zeros(length(lx),1);
fyVec = zeros(length(ly),1);

xp     = param(1);
yp     = param(2);
c      = param(3); 
numIOP = 3;

uniqueID = unique(EOP_ID);
for n = 1:length(uniqueID)
    
    omega  = param(numIOP+6*n-5);
    phi    = param(numIOP+6*n-4);
    kappa  = param(numIOP+6*n-3);
    Xo     = param(numIOP+6*n-2);
    Yo     = param(numIOP+6*n-1);
    Zo     = param(numIOP+6*n-0);
    
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
    
    T = [Xo; Yo; Zo];
    
    I = find(EOP_ID == n);
    fx = zeros(length(I),1);
    fy = zeros(length(I),1);
    for m=1:length(I)
        XYZ = [X(I(m)); Y(I(m)); Z(I(m))];      
        temp = M * (XYZ - T);

        x_img = xp + -c * temp(1) / temp(3);
        y_img = yp + -c * temp(2) / temp(3);
        
        fx(m) = x_img - lx(I(m));
        fy(m) = y_img + ly(I(m));
    end
    
    fxVec(I) = fx;
    fyVec(I) = fy;
    
end
v = [fxVec; fyVec];
end

% unknowns params u x 1
function [v] = stereographicResection(param, X, Y, Z, lx, ly)

omega  = param(1);
phi    = param(2);
kappa  = param(3);
Xo     = param(4);
Yo     = param(5);
Zo     = param(6);
C = param(7);

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

T = [Xo; Yo; Zo];

fx = zeros(length(X),1);
fy = zeros(length(X),1);
for m=1:length(X)
    XYZ = [X(m); Y(m); Z(m)];
    XYZ_s = M * (XYZ - T);
    
    %         lambda = radius / sqrt(XYZ_s(1)*XYZ_s(1) + XYZ_s(2)*XYZ_s(2) + XYZ_s(3)*XYZ_s(3));
    %
    %         XYZ_c = lambda * XYZ_s;
    %
    %         % stereographic projection of point on sphere onto image place
    %         x_img = (radius + cx)/(radius - XYZ_c(3)) * XYZ_c(1);
    %         y_img = (radius + cy)/(radius - XYZ_c(3)) * XYZ_c(2);
    
    d = sqrt(XYZ_s(1)*XYZ_s(1) + XYZ_s(2)*XYZ_s(2) + XYZ_s(3)*XYZ_s(3));
    
    % stereographic projection of point on sphere onto image place
    x_img = ( (2*C)/(d - XYZ_s(3)) )* XYZ_s(1);
    y_img = ( (2*C)/(d - XYZ_s(3)) )* XYZ_s(2);
    
    fx(m) = x_img - lx(m);
    fy(m) = y_img + ly(m);
    %         dx = [dx, x_img - imagePoints_original(m,1)];
    %         dy = [dy, y_img + imagePoints_original(m,2)];
end
v = [fx; fy;];
end

% unknowns params u x 1
function [v] = stereographicResectionCalibration(param, X, Y, Z, lx, ly, EOP_ID)

fxVec = zeros(length(lx),1);
fyVec = zeros(length(ly),1);

xp     = param(1);
yp     = param(2);
C      = param(3); % this is defined as C = radius + c
%     radius = param(4);
numIOP = 3;

uniqueID = unique(EOP_ID);
for n = 1:length(uniqueID)
    
    omega  = param(numIOP+6*n-5);
    phi    = param(numIOP+6*n-4);
    kappa  = param(numIOP+6*n-3);
    Xo     = param(numIOP+6*n-2);
    Yo     = param(numIOP+6*n-1);
    Zo     = param(numIOP+6*n-0);
    
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
    
    T = [Xo; Yo; Zo];
    
    
    I = find(EOP_ID == n);
    fx = zeros(length(I),1);
    fy = zeros(length(I),1);
    for m=1:length(I)
        XYZ = [X(I(m)); Y(I(m)); Z(I(m))];
        XYZ_s = M * (XYZ - T);
        
        %             lambda = radius / sqrt(XYZ_s(1)*XYZ_s(1) + XYZ_s(2)*XYZ_s(2) + XYZ_s(3)*XYZ_s(3));
        %
        %             XYZ_c = lambda * XYZ_s;
        %
        %             % stereographic projection of point on sphere onto image place
        %             x_img = (radius + c)/(radius - XYZ_c(3)) * XYZ_c(1);
        %             y_img = (radius + c)/(radius - XYZ_c(3)) * XYZ_c(2);
        
        d = sqrt(XYZ_s(1)*XYZ_s(1) + XYZ_s(2)*XYZ_s(2) + XYZ_s(3)*XYZ_s(3));
        
        % stereographic projection of point on sphere onto image place
        x_img = xp + ( 2*C/(d - XYZ_s(3)) )* XYZ_s(1);
        y_img = yp + ( 2*C/(d - XYZ_s(3)) )* XYZ_s(2);
        
        fx(m) = x_img - lx(I(m));
        fy(m) = y_img + ly(I(m));
        %         dx = [dx, x_img - imagePoints_original(m,1)];
        %         dy = [dy, y_img + imagePoints_original(m,2)];
    end
    
    fxVec(I) = fx;
    fyVec(I) = fy;
    
end
v = [fxVec; fyVec];
end

% unknowns params u x 1
function [v] = stereographicResectionCalibrationAP(param, X, Y, Z, lx, ly, EOP_ID)

k1     = 0;
k2     = 0;
k3     = 0;
p1     = 0;
p2     = 0;
a1     = 0;
a2     = 0;

fxVec = zeros(length(lx),1);
fyVec = zeros(length(ly),1);

xp     = param(1);
yp     = param(2);
C      = param(3); % this is defined as C = radius + c or approximately as 2*c
k1     = param(4);
k2     = param(5);
k3     = param(6);
% p1     = param(7);
% p2     = param(8);
% a1     = param(9);
% a2     = param(10);



numIOP = 6; % change this myself depending on the number of unknowns

uniqueID = unique(EOP_ID);
for n = 1:length(uniqueID)
    
    omega  = param(numIOP+6*n-5);
    phi    = param(numIOP+6*n-4);
    kappa  = param(numIOP+6*n-3);
    Xo     = param(numIOP+6*n-2);
    Yo     = param(numIOP+6*n-1);
    Zo     = param(numIOP+6*n-0);
    
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
    
    T = [Xo; Yo; Zo];
    
    I = find(EOP_ID == n);
    fx = zeros(length(I),1);
    fy = zeros(length(I),1);
    for m=1:length(I)
        XYZ = [X(I(m)); Y(I(m)); Z(I(m))];
        XYZ_s = M * (XYZ - T);
        
        %             lambda = radius / sqrt(XYZ_s(1)*XYZ_s(1) + XYZ_s(2)*XYZ_s(2) + XYZ_s(3)*XYZ_s(3));
        %
        %             XYZ_c = lambda * XYZ_s;
        %
        %             % stereographic projection of point on sphere onto image place
        %             x_img = (radius + c)/(radius - XYZ_c(3)) * XYZ_c(1);
        %             y_img = (radius + c)/(radius - XYZ_c(3)) * XYZ_c(2);
        
        d = sqrt(XYZ_s(1)*XYZ_s(1) + XYZ_s(2)*XYZ_s(2) + XYZ_s(3)*XYZ_s(3));      
        
        % camera correction model AP = k1, k2, k3, p1, p2, a1, a2 ...
        APSCALE = 1000.0;
        x_bar = lx(I(m)) / APSCALE; % arbitrary scale for numerical stability
        y_bar = -ly(I(m)) / APSCALE; % arbitrary scale for numerical stability
        r = sqrt(x_bar*x_bar + y_bar*y_bar);
        
        delta_x = x_bar*(k1*r*r+k2*r*r*r*r+k3*r*r*r*r*r*r) + p1*(r*r+2.0*x_bar*x_bar)+2.0*p2*x_bar*y_bar + a1*x_bar+a2*y_bar;
        delta_y = y_bar*(k1*r*r+k2*r*r*r*r+k3*r*r*r*r*r*r) + p2*(r*r+2.0*y_bar*y_bar)+2.0*p1*x_bar*y_bar;
             
        % stereographic projection of point on sphere onto image place
        x_img = xp + ( 2*C/(d - XYZ_s(3)) )* XYZ_s(1) + delta_x;
        y_img = yp + ( 2*C/(d - XYZ_s(3)) )* XYZ_s(2) + delta_y;
        
        fx(m) = x_img - lx(I(m));
        fy(m) = y_img + ly(I(m));
    end
    
    fxVec(I) = fx;
    fyVec(I) = fy;
    
end
v = [fxVec; fyVec];
end

% unknowns params u x 1
function [v] = fisheyeEquidistantResectionCalibrationAP(param, X, Y, Z, lx, ly, EOP_ID)

k1     = 0;
k2     = 0;
k3     = 0;
p1     = 0;
p2     = 0;
a1     = 0;
a2     = 0;

fxVec = zeros(length(lx),1);
fyVec = zeros(length(ly),1);

xp     = param(1);
yp     = param(2);
C      = param(3); % this is defined as C = radius + c or approximately as 2*c
% k1     = param(4);
% k2     = param(5);
% k3     = param(6);
% p1     = param(7);
% p2     = param(8);
% a1     = param(9);
% a2     = param(10);



numIOP = 3; % change this myself depending on the number of unknowns

uniqueID = unique(EOP_ID);
for n = 1:length(uniqueID)
    
    omega  = param(numIOP+6*n-5);
    phi    = param(numIOP+6*n-4);
    kappa  = param(numIOP+6*n-3);
    Xo     = param(numIOP+6*n-2);
    Yo     = param(numIOP+6*n-1);
    Zo     = param(numIOP+6*n-0);
    
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
    
    T = [Xo; Yo; Zo];
    
    I = find(EOP_ID == n);
    fx = zeros(length(I),1);
    fy = zeros(length(I),1);
    for m=1:length(I)
        XYZ = [X(I(m)); Y(I(m)); Z(I(m))];
        XYZ_s = M * (XYZ - T);    
        
        % camera correction model AP = k1, k2, k3, p1, p2, a1, a2 ...
        APSCALE = 1000.0;
        x_bar = lx(I(m)) / APSCALE; % arbitrary scale for numerical stability
        y_bar = -ly(I(m)) / APSCALE; % arbitrary scale for numerical stability
        r = sqrt(x_bar*x_bar + y_bar*y_bar);
        
        delta_x = x_bar*(k1*r*r+k2*r*r*r*r+k3*r*r*r*r*r*r) + p1*(r*r+2.0*x_bar*x_bar)+2.0*p2*x_bar*y_bar + a1*x_bar+a2*y_bar;
        delta_y = y_bar*(k1*r*r+k2*r*r*r*r+k3*r*r*r*r*r*r) + p2*(r*r+2.0*y_bar*y_bar)+2.0*p1*x_bar*y_bar;

        % ISPRS "Validation of geometric models for fisheye lenses" journal paper
        x_img = xp + C*XYZ_s(1)*atan2(sqrt(XYZ_s(1)*XYZ_s(1)+XYZ_s(2)*XYZ_s(2)), -XYZ_s(3)) / sqrt(XYZ_s(1)*XYZ_s(1)+XYZ_s(2)*XYZ_s(2));
        y_img = yp + C*XYZ_s(2)*atan2(sqrt(XYZ_s(1)*XYZ_s(1)+XYZ_s(2)*XYZ_s(2)), -XYZ_s(3)) / sqrt(XYZ_s(1)*XYZ_s(1)+XYZ_s(2)*XYZ_s(2));

        fx(m) = x_img - lx(I(m));
        fy(m) = y_img + ly(I(m));
    end
    
    fxVec(I) = fx;
    fyVec(I) = fy;
    
end
v = [fxVec; fyVec];
end

% unknowns params u x 1
function [v] = incidenceAngle(param, X, Y, Z, lx, ly, EOP_ID)

angle = zeros(length(X),1);

xp     = param(1);
yp     = param(2);
C      = param(3); % this is defined as C = radius + c
%     radius = param(4);
numIOP = 3;
% numIOP = 6;

uniqueID = unique(EOP_ID);
for n = 1:length(uniqueID)
    
    omega  = param(numIOP+6*n-5);
    phi    = param(numIOP+6*n-4);
    kappa  = param(numIOP+6*n-3);
    Xo     = param(numIOP+6*n-2);
    Yo     = param(numIOP+6*n-1);
    Zo     = param(numIOP+6*n-0);
    
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
    
    T = [Xo; Yo; Zo];
    
    
    I = find(EOP_ID == n);
    fx = zeros(length(I),1);
    fy = zeros(length(I),1);
    for m=1:length(I)
        XYZ = [X(I(m)); Y(I(m)); Z(I(m))];
        XYZ_s = M * (XYZ - T);
        
%         d = sqrt(XYZ_s(1)*XYZ_s(1) + XYZ_s(2)*XYZ_s(2) + XYZ_s(3)*XYZ_s(3));
%         
%         angle(I(m)) = acos(-XYZ_s(3) / d);
        
        angle(I(m)) = atan2(sqrt(XYZ_s(1).^2+XYZ_s(2).^2),-XYZ_s(3));
    end
    
end
v = [angle];
end