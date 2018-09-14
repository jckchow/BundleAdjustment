%% Target Segmentation
%  Version: 4
%  Programmer: Jacky Chow
%  Date: November 4, 2009
%  Description: Extracts targets with a user specified radius about a seed
%  point that is updated iteratively to bring it closer to the true
%  centroid
%
%  Modified: January 22, 2010
%      - fixed mistake for not actually using variable "margin"
%  Modified: November 11, 2010
%      - uses KD-tree for searching
%  Modified: January 25, 2011
%      - removes KD-tree so it works on my new PC
%      - performs intensity enhancement and contrast stretching for each
%      target locally rather than globally

close all;
clear all;
clc;

min_pts=10;
intensity_threshold=60; %8 bits
% radius=0.1; %m
% margin=0.035; %m
radius=0.05; %m
margin=radius*0.8; %m

circle_color='black'; % color of the points (either white, points with high intensity or black) to be used for refining the seed point.

%% Read in the *.pts file
[filename, pathname] = uigetfile({'*.*'},'Select Point Cloud File');

in=fopen([pathname, filename],'r');
data=textscan(in,'%f %f %f %f %f %f %f','headerlines',1);
% data=textscan(in,'%f %f %f %f','headerlines',1);

fclose(in);

% Read in the X, Y, and Z coordinates from file
XYZI(:,1)=data{1}; % m
XYZI(:,2)=data{2}; % m
XYZI(:,3)=data{3}; % m
XYZI(:,4)=data{4}; % 8 bit



[filename, pathname] = uigetfile({'*.*'},'Select Seed File');

in=fopen([pathname, filename],'r');
data=textscan(in,'%f %f %f %f','headerlines',0);
fclose(in);

% Read in the X, Y, and Z coordinates from file
seed(:,1)=data{1}; % ID #
seed(:,2)=data{2}; % m
seed(:,3)=data{3}; % m
seed(:,4)=data{4}; % m

num_seeds=length(seed(:,1));



%% Segment the Target

% Build the k-d Tree once from the reference datapoints.
% [temp, temp, reference_tree] = kdtree( XYZI(:,1:3), []);

% remove seeds points that doesn't have a nearby target
remove=[];

for n=1:num_seeds
    I=1;
    iteration = 1;
    L=[];
    
    
    % Iteratively update the seed location to be closer to the centroid
    while (~isempty(I) && iteration<10)
        dist=sqrt((seed(n,2)-XYZI(:,1)).^2+(seed(n,3)-XYZI(:,2)).^2+(seed(n,4)-XYZI(:,3)).^2);
        L=find(dist<radius+margin);
        
%         [ target, dist, L] = kdrangequery( reference_tree, seed(n,2:4), radius+margin);
        % if there is no points near the seed point stop segmentation
%         if (isempty(L))
        if (length(L)<min_pts)
            remove=[remove;n];
            break;
        end
        target=XYZI(L,:);
        
        % Intensity Enhancement
        % alpha needs to be between 0 and 1
        % option is a string that's either calling sin or tan
        
        % convert the intensity to be between 0 and 255
        target(:,4)=(target(:,4)-min(target(:,4))).*255./max(target(:,4)-min(target(:,4)));
        
        alpha= 0.8;
        option='sin';
        
        D=range(target(:,4));
        
        if(option=='sin')
            target(:,4)=(D/2).*(1+(1./sin(alpha.*pi./2)).*sin(alpha.*pi.*(target(:,4)./D-1/2)));
        end
        
        if(option=='tan')
            target(:,4)=(D/2).*(1+(1./tan(alpha.*pi./2)).*tan(alpha.*pi.*(target(:,4)./D-1/2)));
        end
%         
%         figure
%         hist(target(:,4),255);
        
        I=find(target(:,4)<=intensity_threshold);
        black=target(I,:);
        J=find(target(:,4)>intensity_threshold);
        white=target(J,:);
        
%         if strcmp(circle_color,'black')
%             seed(n,2:4)=median(black(:,1:3));
%         else
%             seed(n,2:4)=median(white(:,1:3));
%         end
%         
        iteration=iteration+1;
    end
    
    % if no points near seed point skip to the next point
    if (isempty(L))
        continue;
    end
    
    % Segment the points near the updated seed point
% % %     L=find(dist<radius+0.015);
% % %     target=XYZI(L,:);

    % Plot the target
    I=find(target(:,4)<=intensity_threshold);
    black=target(I,:);
    J=find(target(:,4)>intensity_threshold);
    white=target(J,:);
    
    h=figure;
    plot3(white(:,1), white(:,2), white(:,3),'b.');
    hold on;
    plot3(black(:,1), black(:,2), black(:,3),'g.');
    plot3(seed(n,2), seed(n,3), seed(n,4),'r*');
    hold off;
    xlabel('X (m)');
    ylabel('Y (m)');
    zlabel('Z (m)');
    title('Segmented Target')
    legend('white','black','seed');
    axis equal
    saveas(h,[pathname,'Target_',num2str(seed(n,1)),'.tif'],'tif');

% write point cloud of target to file
out=fopen([pathname,num2str(seed(n,1)),'.jck'],'w');
fprintf(out, '  X[m] \t\t\t Y[m]  \t\t\t Z[m] \n');
fprintf(out, [repmat(' %9.3f \t\t\t', 1, size(target,2)), '\n'], target');
fclose(out);

close all;
end

% output file of improved seed points and removed irrelevant seed points
seed(remove,:)=[];
filename(length(filename)-3:length(filename))=[];
out=fopen([pathname,filename,'_updated.jck'],'w');
fprintf(out, '  Target_ID \t\t\t X[m] \t\t\t Y[m] \t\t\t Z[m] \n');
fprintf(out, [repmat(' %9.3f \t\t\t', 1, size(seed,2)), '\n'], seed');
fclose(out);


disp('Program Successful');