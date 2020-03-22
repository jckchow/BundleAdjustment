function Multiple_Circular_Centroid_Extraction_interpolate8
%% Centroid of Circle Extraction
%  Version 5
%  Programmer: Jacky Chow
%  Date: November 9, 2009
%  Description: Computes the 3D coordinates of the centroid of a circle
%  1. Plane fit to the plane target Ax+By+CZ-D=0
%  2. Threshold to separate the black and white in the target
%  3. Rotate the plane until it is parallel to the Z-axis using the normal
%     vector
%  4. Find edge points by interpolating the intensity to form intensity
%     image and perform Canny edge detector for the edge points
%  5. Find the nearest edge point of the background or outside circle, which
%     is the closest to the edge points of the disk
%  6. Best fit circle (x-a)^2+(y-b)^2-r^2=0
%  7. Centroid in (a, b, D)
%  8. Rotate the coordinates back to its original orientation
%
%  Modified: November 16, 2009
%      - Fixed VCE
%  Modified: January 22, 2009
%      - Outputs the plane parameters "a b c and d" for all targets to a file 
%  Modified: June 4, 2010
%      - Plane fitting using least squares is removed and the blunder
%      detection is now based on if the normal distance is larger than
%      3.29*std(normal_distance) + mean(normal_distance) and not the
%      absoluted normal distance as before.
%  Modified: January 25, 2011
%      - Added Parallel Processing capabilities, so parfor is used and the
%      centroid extraction process is a lot faster in this version
%  Modified: March 21, 2020 v8
%      - Added RANSAC to the circle fitting routine

close all;
clear all;
clc;
T_overall=tic;

% Resample method ('nearest', 'linear', 'cubic')
resample='linear';
% intensity_threshold=60;
intensity_threshold=150;
% radius=0.035; %m
% radius=0.10; %m large targets
radius=0.055; %m large targets
circle_color='black';


%% Read in the *.pts file
[directory_filename, pathname] = uigetfile({'*.*'},'Select Point Cloud Directory File');
in=fopen([pathname, directory_filename],'r');
data=textscan(in,'%s','headerlines',0);
filenames=data{1};
fclose(in);

num_targets=length(filenames);
Output_centroid=zeros(num_targets,12);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mean_centroid=zeros(num_targets,3);
median_centroid=zeros(num_targets,3);
plane_normal=zeros(num_targets,4);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    h=figure;
for num=1:num_targets
    T=tic;
    string = sprintf(' Target # %2i', num);
    disp(string);
    filename=char(filenames(num));
    
    in=fopen([pathname, filename],'r');
%     data=textscan(in,'%f %f %f %f %f %f %f','headerlines',1);
    data=textscan(in,'%f %f %f %f','headerlines',1);
    fclose(in);
    
    % Read in the X, Y, and Z coordinates from file
    XYZI=[data{1}, data{2}, data{3}, data{4}];
    
    
    if length(XYZI(:,1))>5000
        num_pts_reduce=length(XYZI(:,1))-5000;
%         num_pts_reduce=round(num_pts_reduce/length(XYZI(:,1)));
%         remove=unique(randi(length(XYZI(:,1)), num_pts_reduce, 1));
         remove=fix(linspace(1, length(XYZI(:,1)), num_pts_reduce));
        XYZI(remove,:)=[];
    end

    
    %% Intensity Enhancement
    % alpha needs to be between 0 and 1
    % option is a string that's either calling sin or tan
    
    alpha= 0.8;    
    option='non';
    
    D=range(XYZI(:,4));
    
    if(option=='sin')
        XYZI(:,4)=(D/2).*(1+(1./sin(alpha.*pi./2)).*sin(alpha.*pi.*(XYZI(:,4)./D-1/2)));
    end
    
    if(option=='tan')
        XYZI(:,4)=(D/2).*(1+(1./tan(alpha.*pi./2)).*tan(alpha.*pi.*(XYZI(:,4)./D-1/2)));
    end
    %%

    
    [disk, background]=Thresholding(XYZI, circle_color, intensity_threshold);
    
    % Solve for the plane parameters (Ax+By+Cz-D=0) using the MATLAB program
    % and use them as initial approximates
    [ABCD, normal, meanX, Xfit, circle_Z_std]=Plane_Fitting_MATLAB(disk(:,1:3), pathname, filename);
    
    % Solve for the plane parameters and its corresponding standard deviation
    % using my plane fitting program
% % % %     [ABCD, ABCD_std]=Plane_Fitting_JC(ABCD,disk(:,1:3), pathname, filename);
    
    [circle, circle_Z, omega, kappa]=Rotate_Plane_and_Edge_Detection_interpolate(ABCD, disk, background, XYZI, pathname, filename, resample,intensity_threshold,h, circle_color);
    
%     if circle>500

    % add a RANSAC circle fit routine
    [a, b, mostInliers]=Circle_Fitting_RANSAC(circle, radius, 1000, 0.003, pathname, filename,h);
    
% % % % %     t=1:1:360;
% % % % %     t=t.*pi/180;
% % % % %     circle_x=a+radius.*cos(t);
% % % % %     circle_y=b+radius.*sin(t);
% % % % %     figure
% % % % %     plot(circle_x,circle_y,'g');
% % % % %     hold on;
% % % % %     plot(circle(:,1), circle(:,2),'r.')
% % % % %     plot(circle(mostInliers,1), circle(mostInliers,2),'b.')
% % % % %     hold off;
% % % % %     axis equal
    [a, b, Cx_ab]=Circle_Fitting(circle(mostInliers,:), radius, pathname, filename,h);

    
%     [a, b, Cx_ab]=Circle_Fitting(circle, radius, pathname, filename,h);
 
    [centroid,centroid_C_x,centroid_std,rho]=Reverse_Rotate_Plane(a,b,Cx_ab,circle,circle_Z, circle_Z_std,omega,kappa,disk,background, pathname, filename,h);
    
    T=toc(T);
    %% Create output directory
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    mean_centroid(num,:)=mean(disk(:,1:3));
    median_centroid(num,:)=median(disk(:,1:3));
    plane_normal(num,:)=ABCD;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    foldername='Output_centroid_interpolate\';
    mkdir(pathname,foldername);
    
    out=fopen([pathname,foldername,'output_centroid_interpolate_',filename,'.jck'],'w');
    
    % write statistic to file
    fprintf(out, 'Program: Centroid_Extraction.m\n');
    fprintf(out, 'Programmer: Jacky Chow\n');
    fprintf(out, 'Date:');
    fprintf(out, date);
    fprintf(out, '\nExecution Time (sec): %f', T);
    fprintf(out, '\nInput Directory: %s\n', pathname);
    fprintf(out, 'Input Filename: %s\n', filename);
    fprintf(out, '\n==========================\n\n');
    % write unknowns to file
    fprintf(out, 'UNKNOWNS: Centroid of Target\n');
    fprintf(out, '=======================================================================\n\n');
    fprintf(out, '  X [m]    \t\t\t\t\t     Y [m]   \t\t\t\t\t    Z [m]\n\n');
    fprintf(out, '  %9.9f    \t\t\t\t\t    %9.9f    \t\t\t\t\t    %9.9f\n\n', centroid);
    fprintf(out, '  std_X [mm]    \t\t\t\t\t     std_Y [mm]   \t\t\t\t\t    std_Z [mm]\n\n');
    fprintf(out, '  %9.9f    \t\t\t\t\t    %9.9f    \t\t\t\t\t    %9.9f\n\n', abs(centroid_std.*1000));
    fprintf(out, 'CORRELATION: Centroid of Target\n');
    fprintf(out, '=======================================================================\n\n');
    fprintf(out, [repmat(' %9.9f \t\t\t', 1, size(rho,2)), '\n'], rho');
    
    fclose(out);
    
    Output_centroid(num,:)=[centroid', centroid_C_x(1,:), centroid_C_x(2,:), centroid_C_x(3,:)];
    
%     close all;
%     clc;
%     clear data XYZI ABCD normal meanX Xfit ABCD ABCD_std disk background circle circle_Z omega kappa a b Cx_ab centroid centroid_C_x centroid_std rho
    
end
T_overall=toc(T_overall);

% write centroid coordinates of all points to one file
out=fopen([pathname,'output_ALL_centroid_interpolate_',directory_filename,'.jck'],'w');
fprintf(out, 'Execution Time (sec): %f\n', T_overall);
% filename X(m) Y(m) Z(m) first_row_of_C_X second_row_of_C_X third_row_of_C_x
for num=1:num_targets
    fprintf(out, '%s    \t\t\t\t\t    %9.9f    \t\t\t\t\t    %9.9f    \t\t\t\t\t    %9.9f    \t\t\t\t\t    %9.9f    \t\t\t\t\t    %9.9f    \t\t\t\t\t    %9.9f    \t\t\t\t\t    %9.9f    \t\t\t\t\t    %9.9f    \t\t\t\t\t    %9.9f    \t\t\t\t\t    %9.9f    \t\t\t\t\t    %9.9f    \t\t\t\t\t    %9.9f\n', char(filenames(num)),Output_centroid(num,:));
end
fclose(out);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dlmwrite([pathname,'mean_centroid.jck'], mean_centroid,'delimiter','\t','precision','%.9f');
dlmwrite([pathname,'median_centroid.jck'], median_centroid,'delimiter','\t','precision','%.9f');
dlmwrite([pathname,'plane_normals.jck'],plane_normal,'delimiter','\t','precision','%.9f')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Centroid Program Successful');

end






%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% FUNCTIONS %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [disk, background]=Thresholding(XYZI, circle_color, threshold)

%% Target Extraction Based on Hard Thresholding
%  Programmer: Jacky Chow
%  Date: September 18, 2009
%  Description


%% Separate the black circle and white background

tic

if circle_color=='white'
    disk_index=find(XYZI(:,4)<=threshold);
    background_index=find(XYZI(:,4)>threshold);
end

if circle_color=='black'
    disk_index=find(XYZI(:,4)>threshold);
    background_index=find(XYZI(:,4)<=threshold);
end

disk=[XYZI(disk_index,1) , XYZI(disk_index,2) , XYZI(disk_index,3) , XYZI(disk_index,4)];
background=[XYZI(background_index,1) , XYZI(background_index,2) , XYZI(background_index,3) , XYZI(background_index,4)];

% figure
plot3(XYZI(disk_index,1), XYZI(disk_index,2), XYZI(disk_index,3),'g.')
hold on
plot3(XYZI(background_index,1), XYZI(background_index,2), XYZI(background_index,3),'r.')
hold off
xlabel('X (mm)');
ylabel('Y (mm)');
zlabel('Z (mm)');
title('Thresholding results');

% % % % %% Plot the plane and the points
% % % %
% % % %
% % % % figure
% % % % [xgrid,ygrid] = meshgrid(linspace(min(XYZI(:,1)),max(XYZI(:,1)),5), ...
% % % %     linspace(min(XYZI(:,2)),max(XYZI(:,2)),5));
% % % % zgrid = (1/normal(3)) .* (meanX*normal - (xgrid.*normal(1) + ygrid.*normal(2)));
% % % % h = mesh(xgrid,ygrid,zgrid,'EdgeColor',[0 0 0],'FaceAlpha',0);
% % % %
% % % % hold on
% % % % ndisk = length(disk_index);
% % % % X1 = [XYZI(disk_index,1) Xfit(disk_index,1) nan*ones(ndisk,1)];
% % % % X2 = [XYZI(disk_index,2) Xfit(disk_index,2) nan*ones(ndisk,1)];
% % % % X3 = [XYZI(disk_index,3) Xfit(disk_index,3) nan*ones(ndisk,1)];
% % % % plot3(X1',X2',X3','-', XYZI(disk_index,1),XYZI(disk_index,2),XYZI(disk_index,3),'.', 'Color',[0 .7 0]);
% % % %
% % % % nbackground = length(background_index);
% % % % X1 = [XYZI(background_index,1) Xfit(background_index,1) nan*ones(nbackground,1)];
% % % % X2 = [XYZI(background_index,2) Xfit(background_index,2) nan*ones(nbackground,1)];
% % % % X3 = [XYZI(background_index,3) Xfit(background_index,3) nan*ones(nbackground,1)];
% % % % plot3(X1',X2',X3','-', XYZI(background_index,1),XYZI(background_index,2),XYZI(background_index,3),'.', 'Color',[1 0 0]);
% % % %
% % % % hold off
% % % % maxlim = max(abs(XYZI(:)))*1.1;
% % % % axis([-maxlim maxlim -maxlim maxlim -maxlim maxlim]);
% % % % axis square
% % % % view(-23.5,5);

% disp('Thresholding Program Successful')
end





function [ABCD, normal, meanX, Xfit, circle_Z_std]=Plane_Fitting_MATLAB(XYZ, pathname, filename)
%% Plane Fitting
%  Programmer: Jacky Chow
%  Date: June 30, 2009

%  - Plane fit using MATLAB functions
%  - File with 1 headerline, and 4 columns (x, y, z, intensity)
%  - Output from Trimble PointScape 3.2

%% Description
%http://www.mathworks.com/products/statistics/demos.html?file=/products/dem
%os/shipping/stats/orthoregdemo.html

%Statistics Toolbox 7.2
% Fitting an Orthogonal Regression Using Principal Components Analysis
%
% Principal Components Analysis can be used to fit a linear regression that minimizes the perpendicular distances from the data to the fitted model. This is the linear case of what is known as Orthogonal Regression or Total Least Squares, and is appropriate when there is no natural distinction between predictor and response variables, or when all variables are measured with error. This is in contrast to the usual regression assumption that predictor variables are measured exactly, and only the response variable has an error component.
%
% For example, given two data vectors x and y, you can fit a line that minimizes the perpendicular distances from each of the points (x(i), y(i)) to the line. More generally, with p observed variables, you can fit an r-dimensional hyperplane in p-dimensional space (r < p). The choice of r is equivalent to choosing the number of components to retain in PCA. It may be based on prediction error, or it may simply be a pragmatic choice to reduce data to a manageable number of dimensions.
%
% In this example, we fit a plane and a line through some data on three observed variables. It's easy to do the same thing for any number of variables, and for any dimension of model, although visualizing a fit in higher dimensions would obviously not be straightforward.
% Contents
%
%     * Fitting a Plane to 3-D Data
%     * Fitting a Line to 3-D Data
%
% Fitting a Plane to 3-D Data
%
% First, we generate some trivariate normal data for the example. Two of the variables are fairly strongly correlated.
%
% randn('state',0);
% X = mvnrnd([0 0 0], [1 .2 .7; .2 1 0; .7 0 1],50);
% plot3(X(:,1),X(:,2),X(:,3),'bo');
% grid on;
% maxlim = max(abs(X(:)))*1.1;
% axis([-maxlim maxlim -maxlim maxlim -maxlim maxlim]);
% axis square
% view(-23.5,5);
%
% Next, we fit a plane to the data using PCA. The coefficients for the first two principal components define vectors that form a basis for the plane. The third PC is orthogonal to the first two, and its coefficients define the normal vector of the plane.
%
% [coeff,score,roots] = princomp(X);
% basis = coeff(:,1:2)
%
% basis =
%
%     0.7139   -0.1302
%     0.0008   -0.9824
%     0.7003    0.1338
%
%
% normal = coeff(:,3)
%
% normal =
%
%    -0.6880
%     0.1867
%     0.7012
%
%
% That's all there is to the fit. But let's look closer at the results, and plot the fit along with the data.
%
% Because the first two components explain as much of the variance in the data as is possible with two dimensions, the plane is the best 2-D linear approximation to the data. Equivalently, the third component explains the least amount of variation in the data, and it is the error term in the regression. The latent roots (or eigenvalues) from the PCA define the amount of explained variance for each component.
%
% pctExplained = roots' ./ sum(roots)
%
% pctExplained =
%
%     0.6825    0.2235    0.0940
%
%
% The first two coordinates of the principal component scores give the projection of each point onto the plane, in the coordinate system of the plane. To get the coordinates of the fitted points in terms of the original coordinate system, we multiply each PC coefficient vector by the corresponding score, and add back in the mean of the data. The residuals are simply the original data minus the fitted points.
%
% [n,p] = size(X);
% meanX = mean(X,1);
% Xfit = repmat(meanX,n,1) + score(:,1:2)*coeff(:,1:2)';
% residuals = X - Xfit;
%
% The equation of the fitted plane is [x y z]*normal - meanX*normal = 0. The plane passes through the point meanX, and its perpendicular distance to the origin is meanX*normal. The perpendicular distance from each point to the plane, i.e., the norm of the residuals, is the dot product of each centered point with the normal to the plane. The fitted plane minimizes the sum of the squared errors.
%
% error = abs((X - repmat(meanX,n,1))*normal);
% sse = sum(error.^2)
%
% sse =
%
%    11.0788
%
%
% To visualize the fit, we can plot the plane, the original data, and their projection to the plane.
%
% [xgrid,ygrid] = meshgrid(linspace(min(X(:,1)),max(X(:,1)),5), ...
%                          linspace(min(X(:,2)),max(X(:,2)),5));
% zgrid = (1/normal(3)) .* (meanX*normal - (xgrid.*normal(1) + ygrid.*normal(2
% )));
% h = mesh(xgrid,ygrid,zgrid,'EdgeColor',[0 0 0],'FaceAlpha',0);
%
% hold on
% above = (X-repmat(meanX,n,1))*normal > 0;
% below = ~above;
% nabove = sum(above);
% X1 = [X(above,1) Xfit(above,1) nan*ones(nabove,1)];
% X2 = [X(above,2) Xfit(above,2) nan*ones(nabove,1)];
% X3 = [X(above,3) Xfit(above,3) nan*ones(nabove,1)];
% plot3(X1',X2',X3','-', X(above,1),X(above,2),X(above,3),'o', 'Color',[0 .7 0
% ]);
% nbelow = sum(below);
% X1 = [X(below,1) Xfit(below,1) nan*ones(nbelow,1)];
% X2 = [X(below,2) Xfit(below,2) nan*ones(nbelow,1)];
% X3 = [X(below,3) Xfit(below,3) nan*ones(nbelow,1)];
% plot3(X1',X2',X3','-', X(below,1),X(below,2),X(below,3),'o', 'Color',[1 0 0]
% );
%
% hold off
% maxlim = max(abs(X(:)))*1.1;
% axis([-maxlim maxlim -maxlim maxlim -maxlim maxlim]);
% axis square
% view(-23.5,5);
%
% Green points are above the plane, red points are below.
% Fitting a Line to 3-D Data
%
% Fitting a straight line to the data is even simpler, and because of the nesting property of PCA, we can use the components that have already been computed. The direction vector that defines the line is given by the coefficients for the first principal component. The second and third PCs are orthogonal to the first, and their coefficients define directions that are perpendicular to the line. The equation of the line is meanX + t*dirVect.
%
% dirVect = coeff(:,1)
%
% dirVect =
%
%     0.7139
%     0.0008
%     0.7003
%
%
% The first coordinate of the principal component scores gives the projection of each point onto the line. As with the 2-D fit, the PC coefficient vectors multiplied by the scores the gives the fitted points in the original coordinate system.
%
% Xfit1 = repmat(meanX,n,1) + score(:,1)*coeff(:,1)';
%
% Plot the line, the original data, and their projection to the line.
%
% t = [min(score(:,1))-.2, max(score(:,1))+.2];
% endpts = [meanX + t(1)*dirVect'; meanX + t(2)*dirVect'];
% plot3(endpts(:,1),endpts(:,2),endpts(:,3),'k-');
%
% X1 = [X(:,1) Xfit1(:,1) nan*ones(n,1)];
% X2 = [X(:,2) Xfit1(:,2) nan*ones(n,1)];
% X3 = [X(:,3) Xfit1(:,3) nan*ones(n,1)];
% hold on
% plot3(X1',X2',X3','b-', X(:,1),X(:,2),X(:,3),'bo');
% hold off
% maxlim = max(abs(X(:)))*1.1;
% axis([-maxlim maxlim -maxlim maxlim -maxlim maxlim]);
% axis square
% view(-23.5,5);
% grid on
%
% While it appears that some of the projections in this plot are not perpendicular to the line, that's just because we're plotting 3-D data in two dimensions. In a live MATLAB(R) figure window, you could interactively rotate the plot to different perspectives to verify that the projections are indeed perpendicular, and to get a better feel for how the line fits the data.
%

tic

% Observation Index
num_pts=length(XYZ(:,1));
obs_index=[1:1:num_pts]';
% use=1:1:num_pts;

%% Outlier Snooping
blunders=1; % just to start the loop
vector_blunders=[]; % stores the normal distance of all the detected blunders
vector_blunders_index=[]; % stores the observations index of the detected blunders

while ~isempty(blunders)
    
    %% Fitting a plane to 3D data following MATLAB
%     [coeff, score, roots] = princomp(XYZ);
%     [coeff, score, roots] = pca(XYZ);
    
    [coeff,score,roots] = pca(XYZ);
    basis = coeff(:,1:2);
    % normal vector of the plane A, B, and C
    normal = coeff(:,3);
    pctExplained = roots' ./ sum(roots);
    
    [n,p] = size(XYZ);
    meanX = mean(XYZ,1);
    distance=meanX*normal;
    
    % Equation of a plane AX+BY+CZ-D=0
    ABCD=[normal; distance];
    
    Xfit = repmat(meanX,n,1) + score(:,1:2)*coeff(:,1:2)';
    % residuals in the X, Y, and Z direction
    residuals = XYZ - Xfit;
    
    % errors is the normal distance between the point and plane
    error = ((XYZ - repmat(meanX,n,1))*normal);
    sse = sum(error.^2);
    
    % Remove index of the outliers
    [max_error, I]=max(abs(error));
    if max_error>3.29*std(error)+mean(error)
        blunders=I;
        vector_blunders=[vector_blunders; max_error];
        vector_blunders_index=[vector_blunders_index; obs_index(I)];
    else
        blunders=[];
    end
    
    XYZ(blunders,:)=[];
    obs_index(blunders)=[];
end

% compute the standard deviation of the depth of the target
circle_Z_std=std(error);

% % % % % % % %% Plot the plane and the points
% % % % % % % figure
% % % % % % % % Green above plane, red below the plane
% % % % % % % [xgrid,ygrid] = meshgrid(linspace(min(XYZ(:,1)),max(XYZ(:,1)),5), ...
% % % % % % %     linspace(min(XYZ(:,2)),max(XYZ(:,2)),5));
% % % % % % % zgrid = (1/normal(3)) .* (meanX*normal - (xgrid.*normal(1) + ygrid.*normal(2)));
% % % % % % % h = mesh(xgrid,ygrid,zgrid,'EdgeColor',[0 0 0],'FaceAlpha',0);
% % % % % % %
% % % % % % % hold on
% % % % % % % above = (XYZ-repmat(meanX,n,1))*normal > 0;
% % % % % % % below = ~above;
% % % % % % % nabove = sum(above);
% % % % % % % X1 = [XYZ(above,1) Xfit(above,1) nan*ones(nabove,1)];
% % % % % % % X2 = [XYZ(above,2) Xfit(above,2) nan*ones(nabove,1)];
% % % % % % % X3 = [XYZ(above,3) Xfit(above,3) nan*ones(nabove,1)];
% % % % % % % plot3(X1',X2',X3','-', XYZ(above,1),XYZ(above,2),XYZ(above,3),'.', 'Color',[0 .7 0]);
% % % % % % % nbelow = sum(below);
% % % % % % % X1 = [XYZ(below,1) Xfit(below,1) nan*ones(nbelow,1)];
% % % % % % % X2 = [XYZ(below,2) Xfit(below,2) nan*ones(nbelow,1)];
% % % % % % % X3 = [XYZ(below,3) Xfit(below,3) nan*ones(nbelow,1)];
% % % % % % % plot3(X1',X2',X3','-', XYZ(below,1),XYZ(below,2),XYZ(below,3),'.', 'Color',[1 0 0]);
% % % % % % %
% % % % % % % hold off
% % % % % % % maxlim = max(abs(XYZ(:)))*1.1;
% % % % % % % axis([-maxlim maxlim -maxlim maxlim -maxlim maxlim]);
% % % % % % % axis square
% % % % % % % view(-23.5,5);
% % % % % % % title('Plane Fit Results: GREEN above plane & RED below the plane');

%% Write to text file
foldername='MATLAB_Plane_Fit\';
mkdir(pathname,foldername);
out=fopen([pathname,foldername,'output_plane_fit_MATLAB_',filename,'.jck'],'w');

% write statistic to file
fprintf(out, 'Program: Plane_Fit_MATLAB.m\n');
fprintf(out, 'Programmer: Jacky Chow\n');
fprintf(out, 'Date:');
fprintf(out, date);
fprintf(out, '\nInput Directory: %s\n', pathname);
fprintf(out, 'Input Filename: %s\n', filename);
fprintf(out, '\n==========================\n\n');
fprintf(out, '  Principal Components Analysis f(X,l) = AX+BY+CZ-D = 0\n');
fprintf(out, '  Number of original points                 :  %9i\n', num_pts);
fprintf(out, '  Number of blunders                        :  %9i\n', length(vector_blunders));
fprintf(out, '  Number of used points                     :  %9i\n', num_pts-length(vector_blunders));
fprintf(out, '  Mean normal distance [mm]                 :  %9.6f\n', 1000.*mean(error));
fprintf(out, '  Standard deviation of normal distance [mm]:  +/- %9.6f\n', 1000.*std(error));
fprintf(out, '  Mean residual in X [mm]                   :  %9.6f\n', 1000.*mean(residuals(:,1)));
fprintf(out, '  Standard deviation of X [mm]              :  +/- %9.6f\n', 1000.*std(residuals(:,1)));
fprintf(out, '  Mean residual in Y [mm]                   :  %9.6f\n', 1000.*mean(residuals(:,2)));
fprintf(out, '  Standard deviation of Y [mm]              :  +/- %9.6f\n', 1000.*std(residuals(:,2)));
fprintf(out, '  Mean residual in Z [mm]                   :  %9.6f\n', 1000.*mean(residuals(:,3)));
fprintf(out, '  Standard deviation of Z [mm]              :  +/- %9.6f\n\n', 1000.*std(residuals(:,3)));

% write unknowns to file
fprintf(out, 'UNKNOWNS: estimated plane parameters (A, B, C, D) [m]\n');
fprintf(out, '=======================================================================\n\n');
fprintf(out, '  A    \t\t\t\t\t     B    \t\t\t\t\t    C    \t\t\t\t\t    D \n\n');
fprintf(out, '  %9.9f    \t\t\t    %9.9f    \t\t\t    %9.9f    \t\t\t    %9.9f\n\n', ABCD);

% write observations to file
fprintf(out, 'BLUNDERS: Normal Distances (m)\n');
fprintf(out, '========================================\n\n');
fprintf(out, '  Obs#  \t\t\t  Normal Distance (m)   \n\n');
Output=[vector_blunders_index, 1000.*vector_blunders];
fprintf(out, [repmat(' %9.6f \t\t\t', 1, size(Output,2)), '\n'], Output');

% write observations to file
fprintf(out, 'OBSERVATIONS: Residual and Normal Distances\n');
fprintf(out, '========================================\n\n');
Output_Observations=[obs_index, 1000.*error, 1000.*residuals];
fprintf(out, '  Obs#   \t\t\t   Nomal Distance [mm]   \t\t\t   residual_X [mm]   \t\t\t   residual_Y [mm]   \t\t\t    residual_Z [mm]\n\n');
fprintf(out, [repmat(' %9.6f \t\t\t', 1, size(Output_Observations,2)), '\n'], Output_Observations');
fprintf(out, '\n\n');

fclose(out);

% disp('MATLAB Plane Fitting Program Successful');
% string = sprintf(' Number of Blunders in Plane Fitting: %9.6f', length(vector_blunders));
% disp(string);
% toc
end


function [X, X_std, aposteriori]=Plane_Fitting_JC(ABCD,XYZ, pathname, filename)
%% Plane Fitting Program (normal distances)
%  Programmer: Jacky Chow
%  Date: Septmber 21, 2009
%  Description:
%  - A program that takes in .pts file from PointScape and does the best
%    plane fit
%  - Uses the plane equation Ax+By+Cz-D=0 as the combined model
%  - A = [df/dA   df/dB    df/dC   df/dD]
%  - B = [df/dx1  df/dx2 ...  df/dy1  df/dy2 ... df/dz1  df/dz2]
%  - G = [df/dA   df/dB    df/dC   df/dD]
%  - Observation accuracy for all 3 XYZ is solved for manually using VCE

tic;

%% Read in the *.pts file



Output_Blunders=[];

% Tells which iteration the blunder was removed in and gives the total
% number of blunders removed
iterations_blunder=0;
iterations_VCE=0;

% threshold for the convergence of the adjustment, it is the absolute
% difference between the aposteriori variance factor in two consecutive
% adjustment iterations
threshold=0.0001; %m
threshold_VCE=0.000000001;

%% form the observation matrix
l=[XYZ(:,1); XYZ(:,2); XYZ(:,3)];

num_original_obs=length(l);
num_original_equ=length(l)/3;

% assign an index number to each observation
obs_index=[1:1:length(XYZ(:,1))]';

% observation accuracy for use in VCE
apriori=1;
apriori_X=1;
apriori_Y=1;
apriori_Z=1;


% number of observations
num_obs=length(l);
% number of equations
num_equ=num_obs/3;

% Build the cofactor matrix of the observations;
sigma_l_x=(0.012).*ones(num_equ,1);
sigma_l_y=(0.012).*ones(num_equ,1);
sigma_l_z=(0.012).*ones(num_equ,1);
C_l=spdiags([sigma_l_x.^2; sigma_l_y.^2; sigma_l_z.^2],0,num_obs, num_obs);


%% Least squares adjustment

%% Begin iteration for blunder detection
% initialize the variable blunders for storing all the detected blunders
blunders=[];
% index cooresponding to the index in the original observation that is
% being removed
blunder_index=[];
% stores the maximum standardized residual for every blunder
blunder_max=[];

%% Least squares adjustment
while (1)
    
    iterations_VCE=iterations_VCE+1;
%     string = sprintf('VCE Iteration: %3i',iterations_VCE);
%     disp(string);
    
    % Stores the index of outlier observations detected by data snooping to be
    % removed
    remove_index=1;
    
    while (~isempty(remove_index))
        %% Configuration Parameters
        
        % number of observations
        num_obs=length(l);
        % number of equations
        num_equ=num_obs/3;
        % number of unknowns
        num_unk=4; % A, B, C, and D
        % number of contraints
        num_con=1;
        % redundancy (degrees of freedom)
        redundancy=num_equ-num_unk+num_con;
        % average redundancy number
        redundancy_avg=redundancy/num_obs;
        
        
        
        %% Obtain the inital approximate value of the unknown parameters
        
        X=ABCD;
        
        l_hat=l;
        sigma_old=1;
        iterations_adjustment=1;
        
        
        
        %         C_l=spdiags([apriori_X.*sigma_l_x.^2; apriori_Y.*sigma_l_y.^2; apriori_Z.*sigma_l_z.^2],0,num_obs, num_obs);
        Q_l=(1/apriori).*C_l;
        % Build the weight matrix
        P=inv(Q_l);
        
        % Build the design matrix A
        A=sparse([l(1:num_equ) l(num_equ+1:num_equ*2) l(2*num_equ+1:num_obs) -1.*ones(num_equ,1)]);
        
        % Build the design matrix B
        
        B = spalloc(num_equ,num_obs,num_equ*2);
        B(:,1:num_equ)=X(1).*speye(num_equ);
        B(:,num_equ+1:2*num_equ)=X(2).*speye(num_equ);
        B(:,2*num_equ+1:3*num_equ)=X(3).*speye(num_equ);
        B=sparse(B);
        
        % Contraint equations
        % g = A^2 + B^2 + C^2 = 1
        G = zeros(num_con,num_unk);
        G=[2.*X(1)  2.*X(2)  2.*X(3)  0];
        
        % Misclosure vector of the contraints
        w_g=1-(X(1).^2+X(2).^2+X(3).^2);
        
        % Build the misclosure vector of the observations
        f_xl=zeros(num_equ,1);
        f_xl=X(1).*l_hat(1:num_equ)+X(2).*l_hat(num_equ+1:2*num_equ)+X(3).*l_hat(2*num_equ+1:num_obs)-X(4).*ones(num_equ,1);
        w_b=B*(l_hat-l)-f_xl;
        
        
        
        
        
        
        
        %Cofactor matrix of the misclosure
        Q_w=B*Q_l*B';
        %% Iteration for the adjustment
        while(iterations_adjustment<100)
            
            
            N=[        A                           Q_w              zeros(num_equ,num_con);
                zeros(num_unk,num_unk)              A'                         G';
                G                       zeros(num_con,num_equ)      zeros(num_con,num_con)];
            
            Q=inv(N);
            %     [delta, k_b, k_g]=Q*[w_b; zeros(num_unk,1); w_g];
            temp=Q*[w_b; zeros(num_unk,1); w_g];
            delta=temp(1:num_unk,1);
            k_b=temp(num_unk+1:num_unk+num_equ,1);
            
            
            X=X+delta;
            
            v=Q_l*B'*k_b;
            
            l_hat=l+v;
            
            
            % Build the design matrix A
            A=sparse([l_hat(1:num_equ) l_hat(num_equ+1:num_equ*2) l_hat(2*num_equ+1:num_obs) -1.*ones(num_equ,1)]);
            
            % Build the design matrix B
            B(:,1:num_equ)=X(1).*speye(num_equ);
            B(:,num_equ+1:2*num_equ)=X(2).*speye(num_equ);
            B(:,2*num_equ+1:3*num_equ)=X(3).*speye(num_equ);
            B=sparse(B);
            
            % Contraint equations
            % g = A^2 + B^2 + C^2 = 1
            G=[2.*X(1)  2.*X(2)  2.*X(3)  0];
            
            % Misclosure vector of the contraints
            w_g=1-(X(1).^2+X(2).^2+X(3).^2);
            
            % Build the misclosure vector
            f_xl=X(1).*l_hat(1:num_equ)+X(2).*l_hat(num_equ+1:2*num_equ)+X(3).*l_hat(2*num_equ+1:num_obs)-X(4).*ones(num_equ,1);
            w_b=B*(l_hat-l)-f_xl;
            
            
            aposteriori=((v')*P*v)./(redundancy);
            sigma_new=sqrt(aposteriori);
            
            %Cofactor matrix of the misclosure
            Q_w=B*Q_l*B';
            
            if(abs(sigma_new-sigma_old)<threshold)
                break;
            else
                iterations_adjustment=iterations_adjustment+1;
                sigma_old=sigma_new;
            end
            
        end
        %% End of adjustment
% %         disp('End of Adjustment')
        % compute the variance-covariance matrix of the misclosure
        % C_w=aposteriori.*Q_w;
        
        %% normal distance between plane and point in mm
        normal_distance=1000.*(sqrt((v(1:num_equ)).^2+(v(num_equ+1:num_equ*2)).^2+(v(num_equ*2+1:num_obs)).^2));
        
        % 99.9% confidence interval, if normal distance is further than threshold
        % remove it
        blunder_threshold=mean(normal_distance)+3.290.*std(normal_distance);
        
        blunders_candidate(:,1)=find(normal_distance>blunder_threshold);
        [normal_distance_max,blunder_candidate_index]=max(normal_distance(blunders_candidate(:,1)));
        blunder_max=[blunder_max; normal_distance_max];
        remove_index=blunders_candidate(blunder_candidate_index);
        
        % delete all blunder candidates after the biggest blunder has been
        % identified
        clear blunders_candidate;
        
        if (~isempty(remove_index))
            iterations_blunder=iterations_blunder+1;
            % if blunder is in the X, find corresponding Y and Z and remove the
            % whole point
            % store index of original obs flagged as blunder and will be removed
            blunder_index=[blunder_index; obs_index(remove_index)];
            % remove blunder from the original observations index variable
            % remove the X, Y, and Z coordinate
            remove=[remove_index; remove_index+num_equ; remove_index+(2*num_equ)];
            % Save information of the blunder for display in text file
            % later on
            blunders=[blunders; (l(remove,1))'];
%             string = sprintf(' Iteration %2i: Blunder Detected (%9.6f > %9.6f): %10i \t %9.6f \t %9.6f \t %9.6f', iterations_blunder, normal_distance_max, blunder_threshold, obs_index(remove_index), (l(remove,1))');
%             disp(string);
            obs_index(remove_index)=[];
            l(remove)=[];
            C_l(:,remove)=[];
            C_l(remove,:)=[];
        end
    end
    
    %% Variance Componenet Estimation
    if (abs(aposteriori-apriori)<0.000000001  || iterations_VCE>100)
        break;
    else
        num_blunderless_obs=length(l)/3;
        C_l=(aposteriori/apriori).*C_l;
    end
    
end



%% Reliability Analysis

% global chi square test %
chi_alpha=0.01;
chi2_y=redundancy*aposteriori/apriori;

chi2_threshold1=chi2inv(chi_alpha/2,redundancy);
chi2_threshold2=chi2inv(1-chi_alpha/2,redundancy);

if (chi2_y>chi2_threshold2 || chi2_y<chi2_threshold1)
    chi2_global_test='failed';
%     disp('chi2_global_test=failed')
else
%     disp('chi2_global_test=passed')
    chi2_global_test='passed';
end

%% standard deviation of the unknown parameters (A, B, C, and D)
Q_xx=inv([A'*inv(Q_w)*A     G'
    G            0]);

Q_x=Q_xx(1:num_unk,1:num_unk);

X_std=sigma_new.*sqrt(diag(Q_x));
X_std=full(X_std);





%% structure the observation measure for output
XYZ_hat=zeros(num_blunderless_obs,3);
XYZ_hat(:,1)=l_hat(1:num_blunderless_obs);
XYZ_hat(:,2)=l_hat(num_blunderless_obs+1:num_blunderless_obs*2);
XYZ_hat(:,3)=l_hat(num_blunderless_obs*2+1:num_blunderless_obs*3);

XYZ_hat_v=zeros(num_blunderless_obs,3);
XYZ_hat_v(:,1)=v(1:num_blunderless_obs);
XYZ_hat_v(:,2)=v(num_blunderless_obs+1:num_blunderless_obs*2);
XYZ_hat_v(:,3)=v(num_blunderless_obs*2+1:num_blunderless_obs*3);

%% Create output directory
foldername='JC_Plane_Fit\';
mkdir(pathname,foldername);
out=fopen([pathname,foldername,'output_plane_fit_JC_normal_distance',filename,'.jck'],'w');

% write statistic to file
fprintf(out, 'Program: Plane_Fit_JC_normal_distance.m\n');
fprintf(out, 'Programmer: Jacky Chow\n');
fprintf(out, 'Date:');
fprintf(out, date);
fprintf(out, '\nInput Directory: %s\n', pathname);
fprintf(out, 'Input Filename: %s\n', filename);
fprintf(out, '\n==========================\n\n');
fprintf(out, '  Conditional Least Squares f(X,l) = AX+BY+CZ-D = 0\n');
fprintf(out, '  Number of raw observations                :  %3i\n', num_original_obs);
fprintf(out, '  Number of raw equations                   :  %3i\n', num_original_equ);
fprintf(out, '  VCE Iteration                             :  %3i\n', iterations_VCE);
fprintf(out, '  Number of Blunders                        :  %3i\n', iterations_blunder);
fprintf(out, '  Number of Adjustment Iterations           :  %3i\n', iterations_adjustment);
fprintf(out, '  Number of observations                    :  %3i\n', num_obs);
fprintf(out, '  Number of unknowns                        :  %3i\n', num_unk);
fprintf(out, '  Number of conditions                      :  %3i\n', num_con);
fprintf(out, '  Number of equations                       :  %3i\n', num_equ);
fprintf(out, '  Percentage of good observations           :  %6.4f\n', (num_obs/num_original_obs)*100);
fprintf(out, '  Redundancy                                :  %3i\n', redundancy);
fprintf(out, '  Average redundancy number                 :  %6.4f\n', redundancy_avg);
fprintf(out, '  SQRT A priori variance factor             :  %6.4f\n', sqrt(apriori));
fprintf(out, '  SQRT of a posteriori variance factor      :  +/- %6.4f\n', sqrt(aposteriori));
fprintf(out, '  Global Chi-Square Test (alpha = %6.4f)    :  %s\n', chi_alpha, chi2_global_test);
fprintf(out, '  Mean residual in X [mm]                   :  %9.6f\n', 1000.*mean(XYZ_hat_v(:,1)));
fprintf(out, '  Standard deviation of X [mm]              :  +/- %9.6f\n', 1000.*std(XYZ_hat_v(:,1)));
fprintf(out, '  Mean residual in Y [mm]                   :  %9.6f\n', 1000.*mean(XYZ_hat_v(:,2)));
fprintf(out, '  Standard deviation of Y [mm]              :  +/- %9.6f\n', 1000.*std(XYZ_hat_v(:,2)));
fprintf(out, '  Mean residual in Z [mm]                   :  %9.6f\n', 1000.*mean(XYZ_hat_v(:,3)));
fprintf(out, '  Standard deviation of Z [mm]              :  +/- %9.6f\n', 1000.*std(XYZ_hat_v(:,3)));
fprintf(out, '  Mean normal distance [mm]                 :  %9.6f\n', mean(normal_distance));
fprintf(out, '  Standard deviation of normal distance [mm]:  +/- %9.6f\n\n', std(normal_distance));

% write unknowns to file
fprintf(out, 'UNKNOWNS: estimated plane parameters (A, B, C, D) and their standard deviations\n');
fprintf(out, '=======================================================================\n\n');
Ouput_Unknowns=[X X_std.*1000];
fprintf(out, '  Plane Parameters [m] \t\t\t Standard Deviation [mm] \n\n');
fprintf(out, [repmat(' %9.9f \t\t\t', 1, size(Ouput_Unknowns,2)), '\n'], Ouput_Unknowns');
fprintf(out, '\n\n');

% write observations to file
fprintf(out, 'OBSERVATIONS: adjusted observation and measures of reliability\n');
fprintf(out, '========================================\n\n');
Output_Observations=[obs_index, normal_distance];
fprintf(out, '  Obs#   \t\t\t   Nomal Distance [mm]\n\n');
fprintf(out, [repmat(' %9.6f \t\t\t', 1, size(Output_Observations,2)), '\n'], Output_Observations');
fprintf(out, '\n\n');

Output_Observations_X=[obs_index    XYZ_hat(:,1)   XYZ_hat_v(:,1).*1000 ];
fprintf(out, '  Obs# \t\t\t  X [m]    \t\t\t    v [mm]  \n\n');
fprintf(out, [repmat(' %9.5f \t\t\t\t', 1, size(Output_Observations_X,2)), '\n'], Output_Observations_X');
fprintf(out, '\n\n');
fprintf(out, '========================================\n\n');
Output_Observations_Y=[obs_index    XYZ_hat(:,2)   XYZ_hat_v(:,2).*1000 ];
fprintf(out, '  Obs# \t\t\t  Y [m]    \t\t\t    v [mm] \n\n');
fprintf(out, [repmat(' %9.5f \t\t\t\t', 1, size(Output_Observations_Y,2)), '\n'], Output_Observations_Y');
fprintf(out, '\n\n');
fprintf(out, '========================================\n\n');
Output_Observations_Z=[obs_index    XYZ_hat(:,3)   XYZ_hat_v(:,3).*1000  ];
fprintf(out, '  Obs# \t\t\t  Z [m]    \t\t\t    v [mm]  \n\n');
fprintf(out, [repmat(' %9.5f \t\t\t\t', 1, size(Output_Observations_Z,2)), '\n'], Output_Observations_Z');
fprintf(out, '\n\n');

% write blunders to file
if (~isempty(blunder_index))
    fprintf(out, 'Detected Blunders (threshold= %6.4f)\n', blunder_threshold);
    fprintf(out, '========================================\n\n');
    fprintf(out, '  Obs#  \t\t\t   X [m]   \t\t\t  Y [m]   \t\t\t  Z [m]    \t\t\t   Normal Distance[mm]\n\n');
    Output_Blunders=[blunder_index,  blunders, blunder_max];
    fprintf(out, [repmat(' %9.5f \t\t\t', 1, size(Output_Blunders,2)), '\n'], Output_Blunders');
else
    fprintf(out, 'No Blunders Detected (threshold= %6.4f)\n', blunder_threshold);
end

fclose(out);

% disp('My Plane Fitting Program Successful')
% toc

end




function [circle_FINAL, circle_Z, omega, kappa]=Rotate_Plane_and_Edge_Detection_interpolate(ABCD,disk, background,XYZI, pathname, filename, resample,threshold,h,circle_color)

%% Rotation of Normal Vector
%  Programmer: Jacky Chow
%  Date: September 22, 2009
%  Description

%% Read in the normal vector of the best fitting plane
% AX+BY+CZ+D=0
foldername='Edge_Detection\';
mkdir(pathname,foldername);

% use if black background
disk(:,4)=255;
background(:,4)=0;
% % use if white background
% disk(:,4)=0;
% background(:,4)=255;

for i=1:length(XYZI(:,4))
    if(XYZI(i,4)<threshold)
        XYZI(i,4)=0;
    else
        XYZI(i,4)=255;
    end
end

[rows, cols]=size(disk);
[background_rows, background_cols]=size(background);
[XYZI_rows, XYZI_cols]=size(XYZI);

%% Rotate to make normal vector parallel to the principal Z-axis
%

disk_XY=zeros(rows,cols-1);
background_XY=zeros(background_rows,background_cols-1);
XYZ=zeros(XYZI_rows, XYZI_cols-1);

% Remove the X component
kappa=atan2(ABCD(1,1),ABCD(2,1));

R3=[cos(kappa) sin(kappa) 0;
    -sin(kappa) cos(kappa) 0;
    0 0 1];


ABCD_X(1:3,1)=R3'*ABCD(1:3);



for i=1:rows
    disk_XY(i,1:3)=(R3'*(disk(i,1:3))')';
end

for i=1:background_rows
    background_XY(i,1:3)=(R3'*(background(i,1:3))')';
end

for i=1:XYZI_rows
    XYZ(i,1:3)=(R3'*(XYZI(i,1:3))')';
end

% Remove the Y component
omega=atan2(ABCD_X(2,1),ABCD_X(3,1));

R1=[1 0 0;
    0 cos(omega) sin(omega);
    0 -sin(omega) cos(omega)];

ABCD_X(1:3,1)=R1'*ABCD_X(1:3);

for i=1:rows
    disk_XY(i,1:3)=(R1'*(disk_XY(i,1:3))')';
end

for i=1:background_rows
    background_XY(i,1:3)=(R1'*(background_XY(i,1:3))')';
end

for i=1:XYZI_rows
    XYZ(i,1:3)=(R1'*(XYZ(i,1:3))')';
end

circle_Z=ABCD_X(3,1).*ABCD(4,1);

%% Resample
% % % % % figure
% % % % % plot(disk_XY(:,1),disk_XY(:,2),'r.');
% % % % % hold on
% % % % % plot(background_XY(:,1),background_XY(:,2),'b.');
% % % % % hold off

%% Convert from cartesian into polar coordinates

% % [theta,alpha,rho] = cart2sph(XYZI(:,1),XYZI(:,2),XYZI(:,3));
% %
% % theta=sort(theta);
% % alpha=sort(alpha);
% %
% % for i=2:length(theta);
% %     theta_diff(i-1)=abs(theta(i)-theta(i-1));
% %     alpha_diff(i-1)=abs(alpha(i)-alpha(i-1));
% % end
% %
% % I=find(theta_diff>mean(theta_diff)+std(theta_diff));
% % J=find(alpha_diff>mean(alpha_diff)+std(alpha_diff));
% %
% % theta_min=min(theta);
% % theta_max=max(theta);
% %
% % alpha_min=min(alpha);
% % alpha_max=max(alpha);
% %
% % theta_range=range(theta);
% % theta_increment=theta_range/length(I);
% %
% % alpha_range=range(alpha);
% % alpha_increment=alpha_range/length(J);
% %
% % clear I J

% % circle_X=sort(XYZI(:,1));
% % circle_Y=sort(XYZI(:,2));

% % circle_X=disk_XY(:,1);
% % circle_Y=disk_XY(:,2);

for i=2:length(XYZ);
    %     X_diff(i-1)=abs(circle_X(i)-circle_X(i-1));
    %     Y_diff_disk(i-1)=abs(disk_XY(i,2)-disk_XY(i-1,2));
    Y_diff_disk(i-1)=abs(XYZ(i,2)-XYZ(i-1,2));
end

% % % for i=2:length(background_XY);
% % % %     X_diff(i-1)=abs(circle_X(i)-circle_X(i-1));
% % %     Y_diff_background(i-1)=abs(background_XY(i,2)-background_XY(i-1,2));
% % % end


I=find(Y_diff_disk>mean(Y_diff_disk)+std(Y_diff_disk));
% % J=find(Y_diff_background>mean(Y_diff_background)+std(Y_diff_background));

for i=2:length(I);
    %     X_diff(i-1)=abs(circle_X(i)-circle_X(i-1));
    I_diff(i-1)=abs(I(i)-I(i-1));
end

I_max=max(I_diff);

% % % % % % % % 
% % % % % % % % [TAR(:,1),TAR(:,2),TAR(:,3)]=cart2sph(XYZ(:,1),XYZ(:,2),XYZ(:,3));
% % % % % % % % 
% % % % % % % % for i=2:length(XYZ);
% % % % % % % %     %     X_diff(i-1)=abs(circle_X(i)-circle_X(i-1));
% % % % % % % %     %     Y_diff_disk(i-1)=abs(disk_XY(i,2)-disk_XY(i-1,2));
% % % % % % % % %     alpha_diff_disk(i-1)=abs(alpha(i,2)-alpha(i-1,2));
% % % % % % % %     if (alpha(i-1,2)-std(alpha(:,2))>alpha(i,2))
% % % % % % % %         I=[I;i];
% % % % % % % %     end
% % % % % % % % end
% % % % % % % % 
% % % % % % % % % % % for i=2:length(background_XY);
% % % % % % % % % % % %     X_diff(i-1)=abs(circle_X(i)-circle_X(i-1));
% % % % % % % % % % %     Y_diff_background(i-1)=abs(background_XY(i,2)-background_XY(i-1,2));
% % % % % % % % % % % end
% % % % % % % % 
% % % % % % % % 
% % % % % % % % I=find(Y_diff_disk>mean(Y_diff_disk)+std(Y_diff_disk));
% % % % % % % % % % J=find(Y_diff_background>mean(Y_diff_background)+std(Y_diff_background));
% % % % % % % % 
% % % % % % % % for i=2:length(I);
% % % % % % % %     %     X_diff(i-1)=abs(circle_X(i)-circle_X(i-1));
% % % % % % % %     I_diff(i-1)=abs(I(i)-I(i-1));
% % % % % % % % end
% % % % % % % % 
% % % % % % % % I_max=max(I_diff);

% X_increment=min(X_diff);
% Y_increment=min(Y_diff);
%
% if X_increment<0.006
%     X_increment=0.006;
% end
%
% if Y_increment<0.006
%     Y_increment=0.006;
% end
%
% X_increment=fix(range(circle_X)/X_increment);
% Y_increment=fix(range(circle_Y)/Y_increment);

% % % circle_X_min=min(disk_XY(:,1));
% % % circle_X_max=max(disk_XY(:,1));
% % %
% % % circle_Y_min=min(disk_XY(:,2));
% % % circle_Y_max=max(disk_XY(:,2));

circle_X_min=min(XYZ(:,1));
circle_X_max=max(XYZ(:,1));

circle_Y_min=min(XYZ(:,2));
circle_Y_max=max(XYZ(:,2));

circle_X_limit=linspace(circle_X_max,circle_X_min,length(I));
circle_Y_limit=linspace(circle_Y_max,circle_Y_min,I_max);

circle_X_limit=linspace(circle_X_max,circle_X_min,100);
circle_Y_limit=linspace(circle_Y_max,circle_Y_min,100);

% %
% % X_increment=circle_X_limit(2)-circle_X_limit(1);
% % Y_increment=circle_Y_limit(2)-circle_Y_limit(1);

% XYZI=[disk_XY, disk(:,4); background_XY, background(:,4)];

intensity_resampled = griddata(XYZ(:,1), XYZ(:,2), XYZI(:,4), circle_X_limit,circle_Y_limit',resample);

% find nan values and replace by zero or 255
[nan_row nan_col]=find(isnan(intensity_resampled));

for i=1:length(nan_row)
    if circle_color == "white"
        intensity_resampled(nan_row(i),nan_col(i))=0;
    else
    %if circle_color == "black"
        intensity_resampled(nan_row(i),nan_col(i))=255;
    end
end

% fill the holes for coded-targets
if circle_color == "black"

    intensity_resampled = imcomplement(intensity_resampled);
    se = strel('disk',5);
    intensity_resampled = imdilate(intensity_resampled,se);
    intensity_resampled = imerode(intensity_resampled,se);
  
%         intensity_resampled = 255 - intensity_resampled;
    % end
    intensity_resampled = imfill(intensity_resampled);
%     intensity_resampled = imbinarize(intensity_resampled);
    intensity_resampled = intensity_resampled + 254;
    J = find(intensity_resampled > 10);
    intensity_resampled(J) = 255;
end

% Plot
% h=figure;
% imagesc(intensity_resampled)
% colormap('gray')
% colorbar
% axis square
% saveas(h,[pathname,'intensity_image_',filename,'.tif'],'tif')

% Canny edge detector and find edge points
edges = edge(intensity_resampled,'canny');

[I J]=find(edges==1);

edges_XY(:,1)=circle_X_limit(J);
edges_XY(:,2)=circle_Y_limit(I);


%% Plot
% h=figure;
plot(disk_XY(:,1),disk_XY(:,2),'c.')
hold on
plot(background_XY(:,1),background_XY(:,2),'g.')
plot(edges_XY(:,1), edges_XY(:,2),'r*');
axis equal
hold off
saveas(h,[pathname,foldername,'Edges_',filename,'.tif'],'tif')

% % % % % 
% % % % % 
% % % % % h=figure;
% % % % % imagesc(intensity_resampled)
% % % % % colormap('gray')
% % % % % colorbar
% % % % % axis square
% % % % % saveas(h,[pathname,foldername,'Edges_Image_',filename,'.tif'],'tif');


%% Plot
% % % % % % % h=figure;
% % % % % % % plot(edges_XY(:,1),edges_XY(:,2),'r*');
% % % % % % % colormap('gray')
% % % % % % % colorbar
% % % % % % % axis square
% % % % % % % saveas(h,[pathname,'Edges',filename,'.tif'],'tif')


circle_FINAL=edges_XY;

% disp('Rotation of Plane Program Successful')
% toc
end

function [a, b, mostInliers]=Circle_Fitting_RANSAC(circle_FINAL, radius, numIterations, maxDist, pathname, filename,h);
%% 2D RANSAC Circle Fitting Program
%  Programmer: Jacky Chow
%  Date: March 21, 2020
%  Description: combined model (x-a)^2+(y-b)^2-r^2=0

%% Read in the circle
tic

foldername='Circle_Fitting_Interpolate\';
mkdir(pathname,foldername);

threshold=0.0001; %m for least squares sigmas
maxSamples = 0;
rng(0,'twister');
mostInliers = [];
inliers = [];
for n=1:numIterations %RANSAC attempts
%     disp(n)
    m = 3; % min number of points to start with for defining a circle
    r = randperm(length(circle_FINAL(:,1)));

    % form the observation matrix
    inliers = r(1:m);

    l=[circle_FINAL(inliers,1);circle_FINAL(inliers,2)];


    % observation accuracy for use in VCE
    apriori=1;
    apriori_X=1;
    apriori_Y=1;

    num_original_obs=length(l);
    num_original_equ=num_original_obs/2;

    % assign an index number to each observation
    obs_index=[1:1:length(circle_FINAL(:,1))]';

    % number of observations
    num_obs=length(l);
    % number of equations
    num_equ=num_obs/2;

    % Build the cofactor matrix of the observations;
    sigma_l_x=(0.012).*ones(num_equ,1);
    sigma_l_y=(0.012).*ones(num_equ,1);
    C_l=diag([sigma_l_x.^2; sigma_l_y.^2]);

    %% Configuration Parameters

    % number of observations
    num_obs=length(l);
    % number of equations
    num_equ=num_obs/2;
    % number of unknowns
    num_unk=2; % centroid of circle a, b
    % number of contraints
    num_con=0;
    % redundancy (degrees of freedom)
    redundancy=num_equ-num_unk+num_con;
    % average redundancy number
    redundancy_avg=redundancy/num_obs;



    %% Obtain the inital approximate value of the unknown parameters

    a=mean(circle_FINAL(:,1));
    b=mean(circle_FINAL(:,2));
    X=[a;b];

    l_hat=l;
    sigma_old=1;
    iterations_adjustment=1;



    %         C_l=diag([apriori_X.*sigma_l_x.^2; apriori_Y.*sigma_l_y.^2]);
    Q_l=(1/apriori).*C_l;
    % Build the weight matrix
    P=inv(Q_l);

    % Build the design matrix A
    A=zeros(num_equ,num_unk);
    A=[-2.*(l(1:num_equ)-X(1)) -2.*(l(num_equ+1:num_obs)-X(2))];

    % Build the design matrix B
    B=zeros(num_equ,num_obs);
    B(:,1:num_equ)=diag(2.*(l(1:num_equ,1)-X(1)));
    B(:,num_equ+1:num_obs)=diag(2.*(l(num_equ+1:num_obs,1)'-X(2)));


    % Build the misclosure vector of the observations
    f_xl=zeros(num_equ,1);
    f_xl=(l(1:num_equ,1)-X(1)).^2+(l(num_equ+1:num_obs,1)-X(2)).^2-radius.^2;
    w_b=B*(l_hat-l)-f_xl;

    %Cofactor matrix of the misclosure
    Q_w=B*Q_l*B';
    %% Iteration for the adjustment
    while(iterations_adjustment<100)


        N=[        A                           Q_w
            zeros(num_unk,num_unk)              A' ];


        Q=inv(N);
        %     [delta, k_b]=Q*[w_b; zeros(num_unk,1)];
        temp=Q*[w_b; zeros(num_unk,1)];
        delta=temp(1:num_unk,1);
        k_b=temp(num_unk+1:num_unk+num_equ,1);


        X=X+delta;

        v=Q_l*B'*k_b;

        l_hat=l+v;


        % Build the design matrix A
        A=[-2.*(l_hat(1:num_equ)-X(1)) -2.*(l_hat(num_equ+1:num_obs)-X(2))];


        % Build the design matrix B
        B=zeros(num_equ,num_obs);
        B(:,1:num_equ)=diag(2.*(l_hat(1:num_equ,1)-X(1)));
        B(:,num_equ+1:num_obs)=diag(2.*(l_hat(num_equ+1:num_obs,1)'-X(2)));


        % Build the misclosure vector
        f_xl=(l_hat(1:num_equ,1)-X(1)).^2+(l_hat(num_equ+1:num_obs,1)-X(2)).^2-radius.^2;
        w_b=B*(l_hat-l)-f_xl;

        aposteriori=((v')*P*v)./(redundancy);
        sigma_new=sqrt(aposteriori);

        %Cofactor matrix of the misclosure
        Q_w=B*Q_l*B';

        if(abs(sigma_new-sigma_old)<threshold)
            break;
        else
            iterations_adjustment=iterations_adjustment+1;
            sigma_old=sigma_new;
        end

    end

    indexOfInliers = logical(zeros(1,length(r)));
    indexOfInliers(1:3) = true;
    for kk=m+1:length(r) % run the RANSAC growing part
        e=sqrt( (circle_FINAL(r(kk),1)-X(1)).^2+(circle_FINAL(r(kk),2)-X(2)).^2 ) -radius;
        
        if (abs(e) < maxDist)
            indexOfInliers(kk) = true;
%             inliers = r(1:kk-1);
%             
%             break
        end
    end
    
    if (sum(indexOfInliers) > length(mostInliers))
        mostInliers = r(indexOfInliers);
    end
        
    
%     disp(length(mostInliers))
end


%% One last LS adjustment with the most inliers to calculate as many inliers as possible
% form the observation matrix
m = length(mostInliers);
inliers = mostInliers;

l=[circle_FINAL(inliers,1);circle_FINAL(inliers,2)];


% observation accuracy for use in VCE
apriori=1;
apriori_X=1;
apriori_Y=1;

num_original_obs=length(l);
num_original_equ=num_original_obs/2;

% assign an index number to each observation
obs_index=[1:1:length(circle_FINAL(:,1))]';

% number of observations
num_obs=length(l);
% number of equations
num_equ=num_obs/2;

% Build the cofactor matrix of the observations;
sigma_l_x=(0.012).*ones(num_equ,1);
sigma_l_y=(0.012).*ones(num_equ,1);
C_l=diag([sigma_l_x.^2; sigma_l_y.^2]);

%% Configuration Parameters

% number of observations
num_obs=length(l);
% number of equations
num_equ=num_obs/2;
% number of unknowns
num_unk=2; % centroid of circle a, b
% number of contraints
num_con=0;
% redundancy (degrees of freedom)
redundancy=num_equ-num_unk+num_con;
% average redundancy number
redundancy_avg=redundancy/num_obs;



%% Obtain the inital approximate value of the unknown parameters

a=mean(circle_FINAL(:,1));
b=mean(circle_FINAL(:,2));
X=[a;b];

l_hat=l;
sigma_old=1;
iterations_adjustment=1;

%         C_l=diag([apriori_X.*sigma_l_x.^2; apriori_Y.*sigma_l_y.^2]);
Q_l=(1/apriori).*C_l;
% Build the weight matrix
P=inv(Q_l);

% Build the design matrix A
A=zeros(num_equ,num_unk);
A=[-2.*(l(1:num_equ)-X(1)) -2.*(l(num_equ+1:num_obs)-X(2))];

% Build the design matrix B
B=zeros(num_equ,num_obs);
B(:,1:num_equ)=diag(2.*(l(1:num_equ,1)-X(1)));
B(:,num_equ+1:num_obs)=diag(2.*(l(num_equ+1:num_obs,1)'-X(2)));


% Build the misclosure vector of the observations
f_xl=zeros(num_equ,1);
f_xl=(l(1:num_equ,1)-X(1)).^2+(l(num_equ+1:num_obs,1)-X(2)).^2-radius.^2;
w_b=B*(l_hat-l)-f_xl;

%Cofactor matrix of the misclosure
Q_w=B*Q_l*B';
%% Iteration for the adjustment
while(iterations_adjustment<100)


    N=[        A                           Q_w
        zeros(num_unk,num_unk)              A' ];


    Q=inv(N);
    %     [delta, k_b]=Q*[w_b; zeros(num_unk,1)];
    temp=Q*[w_b; zeros(num_unk,1)];
    delta=temp(1:num_unk,1);
    k_b=temp(num_unk+1:num_unk+num_equ,1);


    X=X+delta;

    v=Q_l*B'*k_b;

    l_hat=l+v;


    % Build the design matrix A
    A=[-2.*(l_hat(1:num_equ)-X(1)) -2.*(l_hat(num_equ+1:num_obs)-X(2))];


    % Build the design matrix B
    B=zeros(num_equ,num_obs);
    B(:,1:num_equ)=diag(2.*(l_hat(1:num_equ,1)-X(1)));
    B(:,num_equ+1:num_obs)=diag(2.*(l_hat(num_equ+1:num_obs,1)'-X(2)));


    % Build the misclosure vector
    f_xl=(l_hat(1:num_equ,1)-X(1)).^2+(l_hat(num_equ+1:num_obs,1)-X(2)).^2-radius.^2;
    w_b=B*(l_hat-l)-f_xl;

    aposteriori=((v')*P*v)./(redundancy);
    sigma_new=sqrt(aposteriori);

    %Cofactor matrix of the misclosure
    Q_w=B*Q_l*B';

    if(abs(sigma_new-sigma_old)<threshold)
        break;
    else
        iterations_adjustment=iterations_adjustment+1;
        sigma_old=sigma_new;
    end

end

e = sqrt( (circle_FINAL(:,1)-X(1)).^2+(circle_FINAL(:,2)-X(2)).^2) - radius;
indexOfInliers = abs(e) < maxDist;
if (sum(indexOfInliers) > length(mostInliers))
    r = 1:length(circle_FINAL(:,1));
    mostInliers = r(indexOfInliers);
end

end

function [a , b, Cx_ab]=Circle_Fitting(circle_FINAL, radius, pathname, filename,h)

%% 2D Circle Fitting Program
%  Programmer: Jacky Chow
%  Date: Septmber 30, 2009
%  Description: combined model (x-a)^2+(y-b)^2-r^2=0

%% Read in the circle
tic

foldername='Circle_Fitting_Interpolate\';
mkdir(pathname,foldername);




Output_Blunders=[];

% Tells which iteration the blunder was removed in and gives the total
% number of blunders removed
iterations_blunder=0;
iterations_VCE=0;

% threshold for the convergence of the adjustment, it is the absolute
% difference between the aposteriori variance factor in two consecutive
% adjustment iterations
threshold=0.0001; %m
threshold_VCE=0.000000001;

%% form the observation matrix
l=[circle_FINAL(:,1);circle_FINAL(:,2)];

num_original_obs=length(l);
num_original_equ=num_original_obs/2;

% assign an index number to each observation
obs_index=[1:1:length(circle_FINAL(:,1))]';


% observation accuracy for use in VCE
apriori=1;
apriori_X=1;
apriori_Y=1;

% number of observations
num_obs=length(l);
% number of equations
num_equ=num_obs/2;

% Build the cofactor matrix of the observations;
sigma_l_x=(0.012).*ones(num_equ,1);
sigma_l_y=(0.012).*ones(num_equ,1);
C_l=diag([sigma_l_x.^2; sigma_l_y.^2]);

%% Begin iteration for blunder detection
% initialize the variable blunders for storing all the detected blunders
blunders=[];
% index cooresponding to the index in the original observation that is
% being removed
blunder_index=[];
% stores the maximum standardized residual for every blunder
blunder_max=[];

%% Least squares adjustment
while (1)
    
    iterations_VCE=iterations_VCE+1;
%     string = sprintf('VCE Iteration: %3i',iterations_VCE);
%     disp(string);
    
    while (1)
        %% Configuration Parameters
        
        % number of observations
        num_obs=length(l);
        % number of equations
        num_equ=num_obs/2;
        % number of unknowns
        num_unk=2; % centroid of circle a, b
        % number of contraints
        num_con=0;
        % redundancy (degrees of freedom)
        redundancy=num_equ-num_unk+num_con;
        % average redundancy number
        redundancy_avg=redundancy/num_obs;
        
        
        
        %% Obtain the inital approximate value of the unknown parameters
        
        a=mean(circle_FINAL(:,1));
        b=mean(circle_FINAL(:,2));
        X=[a;b];
        
        l_hat=l;
        sigma_old=1;
        iterations_adjustment=1;
        
        
        
        %         C_l=diag([apriori_X.*sigma_l_x.^2; apriori_Y.*sigma_l_y.^2]);
        Q_l=(1/apriori).*C_l;
        % Build the weight matrix
        P=inv(Q_l);
        
        % Build the design matrix A
        A=zeros(num_equ,num_unk);
        A=[-2.*(l(1:num_equ)-X(1)) -2.*(l(num_equ+1:num_obs)-X(2))];
        
        % Build the design matrix B
        B=zeros(num_equ,num_obs);
        B(:,1:num_equ)=diag(2.*(l(1:num_equ,1)-X(1)));
        B(:,num_equ+1:num_obs)=diag(2.*(l(num_equ+1:num_obs,1)'-X(2)));
        
        
        % Build the misclosure vector of the observations
        f_xl=zeros(num_equ,1);
        f_xl=(l(1:num_equ,1)-X(1)).^2+(l(num_equ+1:num_obs,1)-X(2)).^2-radius.^2;
        w_b=B*(l_hat-l)-f_xl;
        
        %Cofactor matrix of the misclosure
        Q_w=B*Q_l*B';
        %% Iteration for the adjustment
        while(iterations_adjustment<100)
            
            
            N=[        A                           Q_w
                zeros(num_unk,num_unk)              A' ];
            
            
            Q=inv(N);
            %     [delta, k_b]=Q*[w_b; zeros(num_unk,1)];
            temp=Q*[w_b; zeros(num_unk,1)];
            delta=temp(1:num_unk,1);
            k_b=temp(num_unk+1:num_unk+num_equ,1);
            
            
            X=X+delta;
            
            v=Q_l*B'*k_b;
            
            l_hat=l+v;
            
            
            % Build the design matrix A
            A=[-2.*(l_hat(1:num_equ)-X(1)) -2.*(l_hat(num_equ+1:num_obs)-X(2))];
            
            
            % Build the design matrix B
            B=zeros(num_equ,num_obs);
            B(:,1:num_equ)=diag(2.*(l_hat(1:num_equ,1)-X(1)));
            B(:,num_equ+1:num_obs)=diag(2.*(l_hat(num_equ+1:num_obs,1)'-X(2)));
            
            
            % Build the misclosure vector
            f_xl=(l_hat(1:num_equ,1)-X(1)).^2+(l_hat(num_equ+1:num_obs,1)-X(2)).^2-radius.^2;
            w_b=B*(l_hat-l)-f_xl;
            
            
            aposteriori=((v')*P*v)./(redundancy);
            sigma_new=sqrt(aposteriori);
            
            %Cofactor matrix of the misclosure
            Q_w=B*Q_l*B';
            
            if(abs(sigma_new-sigma_old)<threshold)
                break;
            else
                iterations_adjustment=iterations_adjustment+1;
                sigma_old=sigma_new;
            end
            
        end
        %% End of adjustment
        % compute the variance-covariance matrix of the misclosure
        % C_w=aposteriori.*Q_w;
        
        %% standard deviation of the unknown parameters (A, B, C, and D)
        Q_x=inv(A'*inv(Q_w)*A);
        
        
        
        X_std=sigma_new.*sqrt(diag(Q_x));
        
        %% Reliability Analysis
        
        % global chi square test %
        chi_alpha=0.01;
        chi2_y=redundancy*aposteriori/apriori;
        
        chi2_threshold1=chi2inv(chi_alpha/2,redundancy);
        chi2_threshold2=chi2inv(1-chi_alpha/2,redundancy);
        
        if (chi2_y>chi2_threshold2 || chi2_y<chi2_threshold1)
            chi2_global_test='failed';
        else
            chi2_global_test='passed';
        end
        
        % compute the cofactor matrix of residuals
        Q22 = Q_x;
        Q12 = inv(Q_w)*A*Q22;
        Q21 = Q12';
        Q11 = inv(Q_w)*(eye(num_equ)-A*Q21);
        Q_v = Q_l*B'*Q11*B*Q_l;
        
        % compute the cofactor matrix of the adjusted observations
        Q_l_hat=Q_l-Q_v;
        
        % compute absorption number
        u_i=Q_l_hat*inv(Q_l);
        u_i=abs(diag(u_i));
        
        % compute the redundancy number
        r_i=Q_v*inv(Q_l);
        r_i=abs(diag(r_i));
        
        % internal reliability (delta_knot_l; minimum detectable blunder) and external reliability (lambda_bar_knot; effect of minimum detectable blunder on the unknown parameters) %
        noncentrality_parameter=4.90; % alpha 1%, beta 1%
        delta_knot_l=sqrt(diag(C_l)).*(noncentrality_parameter)./sqrt(r_i);
        lambda_bar_knot=(noncentrality_parameter^2).*u_i./r_i;
        
        % data snooping using a 99% confidence level
        blunder_alpha=0.01;
        % compute the standardized residuals
        sigma_v=sqrt(apriori).*sqrt(diag(Q_v));
        v_std=abs(v)./sigma_v;
        
        blunder_threshold=norminv(1-blunder_alpha/2, 0, 1);
%         blunder_threshold=10000000000;
        
        blunders_candidate(:,1)=find(v_std>blunder_threshold);
        [v_std_max,blunder_candidate_index]=max(v_std(blunders_candidate(:,1)));
        blunder_max=[blunder_max; v_std_max];
        remove_index=blunders_candidate(blunder_candidate_index);
        
        % delete all blunder candidates after the biggest blunder has been
        % identified
        clear blunders_candidate;
        
        
        if (isempty(remove_index))
             break;
        else
            iterations_blunder=iterations_blunder+1;
            % if blunder is in the X, find corresponding Y and Z and remove the
            % whole point
            if (remove_index<=num_equ)
                % store index of original obs flagged as blunder and will be removed
                blunder_index=[blunder_index; obs_index(remove_index)];
                % remove blunder from the original observations index variable
                % remove the X, Y coordinate
                remove=[remove_index; remove_index+num_equ];
                % Save information of the blunder for display in text file
                % later on
                blunders=[blunders; (l(remove,1))'];
%                 string = sprintf(' Iteration %2i: Blunder in X Detected (%9.6f > %9.6f): %10i \t %9.6f \t %9.6f \t %9.6f', iterations_blunder, v_std_max, blunder_threshold, obs_index(remove_index), (l(remove,1))');
%                 disp(string);
                obs_index(remove_index)=[];
                l(remove)=[];
                C_l(remove,:)=[];
                C_l(:,remove)=[];
            end
            % if blunder is in the Y, find corresponding X and remove the
            % whole point
            if (remove_index>num_equ && remove_index<=(num_equ*2))
                blunder_index=[blunder_index; obs_index(remove_index-num_equ)];
                remove=[remove_index-num_equ; remove_index];
                blunders=[blunders; (l(remove,1))'];
%                 string = sprintf(' Iteration %2i: Blunder in Y Detected (%9.6f > %9.6f): %10i \t %9.6f \t %9.6f \t %9.6f', iterations_blunder, v_std_max, blunder_threshold, obs_index(remove_index-num_equ), (l(remove,1))');
%                 disp(string);
                obs_index(remove_index-num_equ)=[];
                l(remove)=[];
                C_l(remove,:)=[];
                C_l(:,remove)=[];
            end
        end
        
        
    end
    
    %% Store blunders
    if (~isempty(blunder_index))
        Output_Blunders=[Output_Blunders; blunder_index,  blunders, blunder_max];
    end
    
    %% Variance Componenet Estimation
    
    num_blunderless_obs=length(l)/2;
    aposteriori_X=((v(1:num_blunderless_obs)')*P(1:num_blunderless_obs,1:num_blunderless_obs)*v(1:num_blunderless_obs))./sum(r_i(1:num_blunderless_obs));
    aposteriori_Y=((v(num_blunderless_obs+1:num_blunderless_obs*2)')*P(num_blunderless_obs+1:num_blunderless_obs*2,num_blunderless_obs+1:num_blunderless_obs*2)*v(num_blunderless_obs+1:num_blunderless_obs*2))./sum(r_i(num_blunderless_obs+1:num_blunderless_obs*2));
    
    if (abs(aposteriori-apriori)<0.000000001  || iterations_VCE>100)
        break;
    else
        C_l(1:num_blunderless_obs,1:num_blunderless_obs)=(aposteriori_X/apriori_X).*C_l(1:num_blunderless_obs,1:num_blunderless_obs);
        C_l(num_blunderless_obs+1:num_blunderless_obs*2,num_blunderless_obs+1:num_blunderless_obs*2)=(aposteriori_Y/apriori_Y).*C_l(num_blunderless_obs+1:num_blunderless_obs*2,num_blunderless_obs+1:num_blunderless_obs*2);
        
        
    end
    
end
%% structure the observation measure for output
XY_hat=zeros(num_blunderless_obs,3);
XY_hat(:,1)=l_hat(1:num_blunderless_obs);
XY_hat(:,2)=l_hat(num_blunderless_obs+1:num_blunderless_obs*2);

XY_hat_v=zeros(num_blunderless_obs,3);
XY_hat_v(:,1)=v(1:num_blunderless_obs);
XY_hat_v(:,2)=v(num_blunderless_obs+1:num_blunderless_obs*2);

XY_hat_r_i=zeros(num_blunderless_obs,3);
XY_hat_r_i(:,1)=r_i(1:num_blunderless_obs);
XY_hat_r_i(:,2)=r_i(num_blunderless_obs+1:num_blunderless_obs*2);


XY_hat_delta_knot_l=zeros(num_blunderless_obs,3);
XY_hat_delta_knot_l(:,1)=delta_knot_l(1:num_blunderless_obs);
XY_hat_delta_knot_l(:,2)=delta_knot_l(num_blunderless_obs+1:num_blunderless_obs*2);


XY_hat_lambda_bar_knot=zeros(num_blunderless_obs,3);
XY_hat_lambda_bar_knot(:,1)=lambda_bar_knot(1:num_blunderless_obs);
XY_hat_lambda_bar_knot(:,2)=lambda_bar_knot(num_blunderless_obs+1:num_blunderless_obs*2);


%% End of computation


t=1:1:360;
t=t.*pi/180;
% x = a + r*cos(t)
% y = b + r*sin(t)
circle_x=X(1)+radius.*cos(t);
circle_y=X(2)+radius.*sin(t);

% h=figure;
plot(circle_x,circle_y,'g');
hold on
plot(circle_FINAL(:,1),circle_FINAL(:,2),'r.');
plot(X(1),X(2),'g+');
plot(l(1:num_blunderless_obs),l(num_blunderless_obs+1:2*num_blunderless_obs),'b.');
plot(l_hat(1:num_blunderless_obs),l_hat(num_blunderless_obs+1:2*num_blunderless_obs),'c.');
hold off
axis equal
xlabel('X(m)');
ylabel('Y(m)');
title('Centroid of Circle');
legend('Best Fit Circle','Blunders','Centroid','Edge Points','Adjusted Obs');
saveas(h,[pathname,foldername,'Figure_Circle_',filename,'.tif'],'tif')


% disp('Circle Fitting Program Successful')
% toc

%% Create output directory
out=fopen([pathname,foldername,'output_circle_fitting_interpolate_',filename,'.jck'],'w');

% write statistic to file
fprintf(out, 'Program: Circle_Fitting.m\n');
fprintf(out, 'Programmer: Jacky Chow\n');
fprintf(out, 'Date:');
fprintf(out, date);
fprintf(out, '\nInput Directory: %s\n', pathname);
fprintf(out, 'Input Filename: %s\n', filename);
fprintf(out, '\n==========================\n\n');
fprintf(out, '  Conditional Least Squares f(X,l) = (x-a)^2+(y-b)^2-r^2 = 0\n');
fprintf(out, '  Number of raw observations                :  %3i\n', num_original_obs);
fprintf(out, '  Number of raw equations                   :  %3i\n', num_original_equ);
fprintf(out, '  VCE Iteration                             :  %3i\n', iterations_VCE);
fprintf(out, '  Number of blunders                        :  %3i\n', iterations_blunder);
fprintf(out, '  Number of adjustment iterations           :  %3i\n', iterations_adjustment);
fprintf(out, '  Number of blunderless observations        :  %3i\n', num_obs);
fprintf(out, '  Number of unknowns                        :  %3i\n', num_unk);
fprintf(out, '  Number of conditions                      :  %3i\n', num_con);
fprintf(out, '  Number of equations                       :  %3i\n', num_equ);
fprintf(out, '  Percentage of good observations           :  %6.4f\n', (num_obs/num_original_obs)*100);
fprintf(out, '  Redundancy                                :  %3i\n', redundancy);
fprintf(out, '  Average redundancy number                 :  %6.4f\n', redundancy_avg);
fprintf(out, '  SQRT A priori variance factor             :  %6.4f\n', sqrt(apriori));
fprintf(out, '  SQRT of a priori variance factor for X    :  +/- %6.4f\n', sqrt(apriori_X));
fprintf(out, '  SQRT of a priori variance factor for Y    :  +/- %6.4f\n', sqrt(apriori_Y));
fprintf(out, '  SQRT of a posteriori variance factor      :  +/- %6.4f\n', sqrt(aposteriori));
fprintf(out, '  SQRT of a posteriori variance factor for X:  +/- %6.4f\n', sqrt(aposteriori_X));
fprintf(out, '  SQRT of a posteriori variance factor for Y:  +/- %6.4f\n', sqrt(aposteriori_Y));
fprintf(out, '  Global Chi-Square Test (alpha = %6.4f)    :  %s\n', chi_alpha, chi2_global_test);
fprintf(out, '  Mean residual in X [mm]                   :  %9.6f\n', 1000.*mean(XY_hat_v(:,1)));
fprintf(out, '  Standard deviation of X [mm]              :  +/- %9.6f\n', 1000.*std(XY_hat_v(:,1)));
fprintf(out, '  Mean residual in Y [mm]                   :  %9.6f\n', 1000.*mean(XY_hat_v(:,2)));
fprintf(out, '  Standard deviation of Y [mm]              :  +/- %9.6f\n', 1000.*std(XY_hat_v(:,2)));

% write unknowns to file
fprintf(out, 'UNKNOWNS: Centroid of Circle (a, b) and Standard Deviation\n');
fprintf(out, '=======================================================================\n\n');
Ouput_Unknowns=[X X_std.*1000];
fprintf(out, '  Centroid [m] \t\t\t Standard Deviation [mm] \n\n');
fprintf(out, [repmat(' %9.9f \t\t\t', 1, size(Ouput_Unknowns,2)), '\n'], Ouput_Unknowns');
fprintf(out, '\n\n');

% write observations to file
fprintf(out, 'OBSERVATIONS: adjusted observation and measures of reliability\n');
fprintf(out, '========================================\n\n');

Output_Observations_X=[obs_index    XY_hat(:,1)   XY_hat_v(:,1).*1000    XY_hat_r_i(:,1)    XY_hat_delta_knot_l(:,1)    XY_hat_lambda_bar_knot(:,1) ];
fprintf(out, '  Obs# \t\t\t  X [m]    \t\t\t    v [mm]  \t\t\t   r_i     \t\t\t    Internal Reliability   \t\t\t    External Reliability\n\n');
fprintf(out, [repmat(' %9.5f \t\t\t\t', 1, size(Output_Observations_X,2)), '\n'], Output_Observations_X');
fprintf(out, '\n\n');
fprintf(out, '========================================\n\n');
Output_Observations_Y=[obs_index    XY_hat(:,2)   XY_hat_v(:,2).*1000    XY_hat_r_i(:,2)    XY_hat_delta_knot_l(:,2)    XY_hat_lambda_bar_knot(:,2) ];
fprintf(out, '  Obs# \t\t\t  Y [m]    \t\t\t    v [mm]  \t\t\t   r_i     \t\t\t    Internal Reliability   \t\t\t    External Reliability\n\n');
fprintf(out, [repmat(' %9.5f \t\t\t\t', 1, size(Output_Observations_Y,2)), '\n'], Output_Observations_Y');
fprintf(out, '\n\n');

% write blunders to file
if (~isempty(Output_Blunders))
    fprintf(out, 'Baardas Data Snooping: Detected Blunders (alpha = %6.4f, threshold= %6.4f)\n', blunder_alpha, blunder_threshold);
    fprintf(out, '========================================\n\n');
    fprintf(out, '  Obs#  \t\t\t   X [m]   \t\t\t  Y [m]    \t\t\t   Standardized Residual\n\n');
    Output_Blunders=[blunder_index,  blunders, blunder_max];
    fprintf(out, [repmat(' %9.5f \t\t\t', 1, size(Output_Blunders,2)), '\n'], Output_Blunders');
else
    fprintf(out, 'Baardas Data Snooping: No Blunders Detected (alpha = %6.4f, threshold= %6.4f)\n', blunder_alpha, blunder_threshold);
end

fclose(out);

a=X(1);
b=X(2);
Cx_ab=aposteriori*Q_x;
end

function [centroid,C_x,centroid_std,rho]=Reverse_Rotate_Plane(a,b,Cx_ab,circle_FINAL,circle_Z,circle_Z_std,omega,kappa,disk,background,pathname,filename,h)
%% Reverse Rotation of Normal Vector
%  Programmer: Jacky Chow
%  Date: September 22, 2009
%  Description


%% Rotate to make normal vector parallel to the principal Z-axis
%

foldername='Centroid_Interpolate\';
mkdir(pathname,foldername);

tic

centroid=[a; b; circle_Z];
% centroid_std=[a_std; b_std; circle_Z_std];

kappa=-kappa;

omega=-omega;

R3=[cos(kappa) sin(kappa) 0;
    -sin(kappa) cos(kappa) 0;
    0 0 1];

R1=[1 0 0;
    0 cos(omega) sin(omega);
    0 -sin(omega) cos(omega)];

centroid=R1'*centroid;
centroid=R3'*centroid;


% Error Propagation of direct model
% X=f(l)
J=R3'*R1';
C_l=zeros(3,3);
C_l(1:2,1:2)=Cx_ab;
C_l(3,3)=circle_Z_std.^2;
C_x=J*C_l*J';
centroid_std=sqrt(diag(C_x));

% compute correlation matrix
[n,m]=size(C_x);
rho=zeros(n,m);
for i=1:n
    for j=1:m
        rho(i,j)=C_x(i,j)/(sqrt(C_x(i,i))*sqrt(C_x(j,j)));
    end
end

% h=figure;
plot3(disk(:,1), disk(:,2), disk(:,3),'c.');
hold on
plot3(background(:,1), background(:,2), background(:,3),'y.');
plot3(centroid(1,1), centroid(2,1), centroid(3,1),'r*');
hold off;
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
axis equal;
title('Centroid of Target');
saveas(h,[pathname,foldername,'Figure_Centroid_',filename,'.tif'],'tif')

% disp('Reverse Rotated Plane Program Successful');
% toc
end
