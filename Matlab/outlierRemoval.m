%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Program: Outlier Removal
%%% Programmer: Jacky Chow
%%% Date: April 25, 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

%% Input parameters

%%%%%% Go Pro
% inputImageFilename   = "C:\Users\jckch\OneDrive - University of Calgary\Google Drive\UbuntuVirtualShared\bundleAdjustment\omnidirectionalCamera\gopro_2020_04_01\gopro_screened_manual (copy).pho";
% inputOutlierFilename = "C:\Users\jckch\OneDrive - University of Calgary\Google Drive\UbuntuVirtualShared\bundleAdjustment\omnidirectionalCamera\gopro_2020_04_01\outlierList.txt";
% outputImageFilename  = "C:\Users\jckch\OneDrive - University of Calgary\Google Drive\UbuntuVirtualShared\bundleAdjustment\omnidirectionalCamera\gopro_2020_04_01\gopro_screened_manualOutlierRemoval.pho";

% % Training GoPro
inputImageFilename   = "C:\Users\jckch\OneDrive - University of Calgary\Google Drive\UbuntuVirtualShared\bundleAdjustment\omnidirectionalCamera\gopro_2020_04_01\TrainingTesting\goproTraining.pho";
inputOutlierFilename = "C:\Users\jckch\OneDrive - University of Calgary\Google Drive\UbuntuVirtualShared\bundleAdjustment\omnidirectionalCamera\gopro_2020_04_01\outlierList.txt";
outputImageFilename  = "C:\Users\jckch\OneDrive - University of Calgary\Google Drive\UbuntuVirtualShared\bundleAdjustment\omnidirectionalCamera\gopro_2020_04_01\TrainingTesting\goproTraining_manualOutlierRemoval1.pho";

% Testing GoPro
% inputImageFilename   = "C:\Users\jckch\OneDrive - University of Calgary\Google Drive\UbuntuVirtualShared\bundleAdjustment\omnidirectionalCamera\gopro_2020_04_01\TrainingTesting\goproTesting.pho";
% inputOutlierFilename = "C:\Users\jckch\OneDrive - University of Calgary\Google Drive\UbuntuVirtualShared\bundleAdjustment\omnidirectionalCamera\gopro_2020_04_01\outlierList.txt";
% outputImageFilename  = "C:\Users\jckch\OneDrive - University of Calgary\Google Drive\UbuntuVirtualShared\bundleAdjustment\omnidirectionalCamera\gopro_2020_04_01\TrainingTesting\goproTesting_manualOutlierRemoval.pho";

image       = load(inputImageFilename);
outlierList = load(inputOutlierFilename);

disp('Outliers to be removed: ')
disp(outlierList)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% read in image file
% ptID, stnID, x, y, xStdDev, yStdDev, xCorr, yCorr
%%% read in list of outliers to be removed
% ptID, stnID, sensorID
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

blunder = zeros(length(outlierList(:,1)),1);

% Find match by station first, then find the matching point there
for iter = 1:length(blunder)
    
    matchStation  = find( image(:,2)==outlierList(iter,2) );
    matchPoint    = find( image(matchStation,1)==outlierList(iter,1) );
    if (~isempty(matchStation(matchPoint)))
        blunder(iter) = matchStation(matchPoint);
    end
    
end

%remove the no matches, this happens when we run the full dataset to remove
%outliers then later use that file to remove points in the training or
%testing set
I = find(blunder == 0);
blunder(I) = [];

% Actual outlier removal
image(blunder,:) = [];

% Output file
dlmwrite(outputImageFilename,image,'delimiter',' ','precision',7);
disp('Program Successful ^-^')