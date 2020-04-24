clear all; close all; clc;

img = load("C:\Users\jckch\OneDrive - University of Calgary\Google Drive\UbuntuVirtualShared\image.jck");
iop = load("C:\Users\jckch\OneDrive - University of Calgary\Google Drive\UbuntuVirtualShared\image.jck");
iop = [1 1914.703001 -1412.368644 1754.695488 1000.000000 1000.000000 1000.000000];

w = [img(:,6)./img(:,10), img(:,7)./img(:,11)];
e = sqrt( w(:,1).^2+w(:,2).^2 );
I = find( e > 3.291);
[val, index] = max(e);
disp(['Max error: ' , num2str(val)])
disp(['  wx,wy: ',num2str(w(index,:))])
disp(['  Point: ',num2str(img(index,1:3))])
disp(['  x,y: ', num2str(img(index,4)), ', ',num2str(img(index,5)) ])

figure;
subplot(1,3,1);
plot(img(:,6), img(:,7), 'g.');
axis equal;
% xlim([-5 5])
% ylim([-5 5])
title('Residuals')

% figure;
subplot(1,3,2);
plot(w(:,1), w(:,2), 'g.');
hold on;
plot(w(I,1), w(I,2), 'b.');
plot(w(index,1), w(index,2), 'r*')
hold off;
axis equal;
legend('Inliers', 'Outliers', 'Max Error')
% xlim([-5 5])
% ylim([-5 5])
title('Normalized Residuals')

% figure;
subplot(1,3,3);
plot(img(:,4), img(:,5), 'g.');
hold on;
plot(img(I,4), img(I,5), 'b.');
plot(img(index,4), img(index,5), 'r*');
hold off;
axis equal;
legend('Inliers', 'Outliers', 'Max Error')
% xlim([-5 5])
% ylim([-5 5])
title('Observations')

figure;
subplot(2,3,1)
plot(img(:,4)-iop(2), img(:,6), 'r.')
xlabel('x')
ylabel('v_x')
subplot(2,3,2)
plot(img(:,4)-iop(2), img(:,7), 'r.')
xlabel('x')
ylabel('v_y')
subplot(2,3,4)
plot(img(:,5)-iop(3), img(:,6), 'g.')
xlabel('y')
ylabel('v_x')
subplot(2,3,5)
plot(img(:,5)-iop(3), img(:,7), 'g.')
xlabel('y')
ylabel('v_y')
r = sqrt( (img(:,4)-iop(2)).^2 + (img(:,5)-iop(3)).^2 );
subplot(2,3,3)
plot(r, img(:,6), 'r.')
xlabel('r')
ylabel('v_x')
subplot(2,3,6)
plot(r, img(:,7), 'g.')
xlabel('r')
ylabel('v_y')

vr = dot([img(:,6:7)], [ (img(:,4)-iop(2)), (img(:,5)-iop(3)) ]./r, 2);
figure
plot(r, vr, '.')
xlabel('r')
ylabel('v_r')
