close all; clear all; clc;

image = load('C:\Users\jckch\OneDrive - University of Calgary\Google Drive\UbuntuVirtualShared\omnidirectionalCamera\gopro\conventional_K1K2K3P1P2\image.jck');
iop = load('C:\Users\jckch\OneDrive - University of Calgary\Google Drive\UbuntuVirtualShared\omnidirectionalCamera\gopro\conventional_K1K2K3P1P2\iop.jck');
eop = load('C:\Users\jckch\OneDrive - University of Calgary\Google Drive\UbuntuVirtualShared\omnidirectionalCamera\gopro\conventional_K1K2K3P1P2\EOP.jck');


image2 = load('C:\Users\jckch\OneDrive - University of Calgary\Google Drive\UbuntuVirtualShared\omnidirectionalCamera\gopro\new\AfterKNNIOPK1K2K3\image.jck');
iop2 = load('C:\Users\jckch\OneDrive - University of Calgary\Google Drive\UbuntuVirtualShared\omnidirectionalCamera\gopro\new\AfterKNNIOPK1K2K3\iop.jck');
eop2 = load('C:\Users\jckch\OneDrive - University of Calgary\Google Drive\UbuntuVirtualShared\omnidirectionalCamera\gopro\new\AfterKNNIOPK1K2K3\EOP.jck');

cost = load('C:\Users\jckch\OneDrive - University of Calgary\Google Drive\UbuntuVirtualShared\omnidirectionalCamera\nikon\AfterkNN\costs.jck');

image3 = load('C:\Users\jckch\OneDrive - University of Calgary\Google Drive\UbuntuVirtualShared\omnidirectionalCamera\gopro\beforeCalibration\image.jck');

image4 = load('C:\Users\jckch\OneDrive - University of Calgary\Google Drive\UbuntuVirtualShared\omnidirectionalCamera\nikon\Conventional_K1K2\image.jck');
iop4 = load('C:\Users\jckch\OneDrive - University of Calgary\Google Drive\UbuntuVirtualShared\omnidirectionalCamera\nikon\Conventional_K1K2\iop.jck');
eop4 = load('C:\Users\jckch\OneDrive - University of Calgary\Google Drive\UbuntuVirtualShared\omnidirectionalCamera\nikon\Conventional_K1K2\EOP.jck');

image5 = load('C:\Users\jckch\OneDrive - University of Calgary\Google Drive\UbuntuVirtualShared\omnidirectionalCamera\nikon\AfterkNN\image.jck');
iop5 = load('C:\Users\jckch\OneDrive - University of Calgary\Google Drive\UbuntuVirtualShared\omnidirectionalCamera\nikon\AfterkNN\iop.jck');
eop5 = load('C:\Users\jckch\OneDrive - University of Calgary\Google Drive\UbuntuVirtualShared\omnidirectionalCamera\nikon\AfterkNN\EOP.jck');


%% plot the residuals 
figure;
quiver(image(:,2),image(:,3),image(:,4), image(:,5),'m.')
xlabel('x image residuals [pix]')
ylabel('y image residuals [pix]')

figure;
plot(image(:,4), image(:,5),'m.')
xlabel('x image residuals [pix]')
ylabel('y image residuals [pix]')

figure;
plot(image2(:,4), image2(:,5),'c.')
xlabel('x image residuals [pix]')
ylabel('y image residuals [pix]')

figure;
plot(image3(:,4), image3(:,5),'r.')
xlabel('x image residuals [pix]')
ylabel('y image residuals [pix]')

figure;
plot(image4(:,4), image4(:,5),'g.')
xlabel('x image residuals [pix]')
ylabel('y image residuals [pix]')

figure;
plot(image5(:,4), image5(:,5),'b.')
xlabel('x image residuals [pix]')
ylabel('y image residuals [pix]')

figure;
histfit(image(:,4),100)
xlabel('x image residuals [pix]')
ylabel('Frequency')
xlim([-5 5])
ylim([0 120])

figure;
histfit(image2(:,4),100)
xlabel('x image residuals [pix]')
ylabel('Frequency')
xlim([-5 5])
ylim([0 120])

figure;
histfit(image3(:,4),100)
xlabel('x image residuals [pix]')
ylabel('Frequency')
xlim([-5 5])
ylim([0 80])

figure;
histfit(image4(:,4),100,'normal')
xlabel('x image residuals [pix]')
ylabel('Frequency')
xlim([-5 5])
ylim([0 80])

figure;
histfit(image5(:,4),100)
xlabel('x image residuals [pix]')
ylabel('Frequency')
xlim([-5 5])
ylim([0 80])

figure;
histfit(image(:,5),100,'kernel')
xlabel('y image residuals [pix]')
ylabel('Frequency')
xlim([-5 5])
ylim([0 120])

figure;
histfit(image2(:,5),100,'normal')
xlabel('y image residuals [pix]')
ylabel('Frequency')
xlim([-5 5])
ylim([0 120])

figure;
histfit(image3(:,5),100,'normal')
xlabel('y image residuals [pix]')
ylabel('Frequency')
xlim([-5 5])
ylim([0 80])

figure;
histfit(image4(:,5),100,'normal')
xlabel('y image residuals [pix]')
ylabel('Frequency')
xlim([-5 5])
ylim([0 80])

figure;
histfit(image5(:,5),100,'normal')
xlabel('y image residuals [pix]')
ylabel('Frequency')
xlim([-5 5])
ylim([0 80])


figure;
histfit([image3(:,4);image3(:,5)],100,'normal')
xlabel('Image residuals [pix]')
ylabel('Frequency')
xlim([-5 5])
ylim([0 200])


figure;
histfit([image4(:,4);image4(:,5)],100,'kernel')
xlabel('Image residuals [pix]')
ylabel('Frequency')
xlim([-5 5])
ylim([0 200])

figure;
plot(1:315, 4.024652675*ones(315),'r')
hold on;
plot(1:315, 0.439281703*ones(315),'b')



figure;
histfit([image5(:,4);image5(:,5)],100,'normal')
xlabel('Image residuals [pix]')
ylabel('Frequency')
xlim([-5 5])
ylim([0 200])


figure;
plot(image(:,2),image(:,4),'m.')

figure;
plot(image(:,2),image(:,5),'m.')

figure;
plot(image(:,3),image(:,4),'m.')

figure;
plot(image(:,3),image(:,5),'m.')

r = sqrt((image(:,2)-iop(2)).^2 + (image(:,3)-iop(3)).^2);
figure;
plot(r, image(:,4),'m.')
xlabel('Radial distance from principal point [pix]')
ylabel('x image residuals [pix]')

r = sqrt((image(:,2)-iop(2)).^2 + (image(:,3)-iop(3)).^2);
figure;
plot(r, image2(:,4),'c.')
xlabel('Radial distance from principal point [pix]')
ylabel('x image residuals [pix]')

figure;
plot(r, image(:,5),'m.')
xlabel('Radial distance from principal point [pix]')
ylabel('y image residuals [pix]')

figure;
plot(r, image2(:,5),'c.')
xlabel('Radial distance from principal point [pix]')
ylabel('y image residuals [pix]')

a = cost(:,1)./cost(:,2);
b = cost(:,3)./cost(:,4);
figure;
plot(1:length(a),a)
xlabel('Number of iterations')
ylabel('A posteriori variance factor')

figure;
plot(1:length(b),b)

