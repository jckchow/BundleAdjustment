# BundleAdjustment

Date: December 21, 2017

1. Run cmake with destination set to a new folder you create called "build"
2. Navigate to the build folder and run "make" to compile
3. Call ./bundleAdjustment to run the code

Note: If the y-axis is set to -1 it means we have a LHS and currently the code doesn't do anything with it. The user need to manually do the following 3 things:

- Change image observation y to -y
- Change yp to -yp
- Change p2 to -p2 (tangential component of the DLD)

Input files:
*.pho has the following headings in order
- Target ID
- Station ID
- x img measurement
- y img measurement
- x measurmeent noise
- y measurement noise
- x correction to be added
- y correction to be added

*.iop has the following headings in order
- Sensor/Camera ID
- y-axis direction (+1 or -1)
- minimum dimension of horizontal pixels
- minimum dimension of vertical pixels
- maximum dimension of horizontal pixels
- maximum dimension of vertical pixels
- xp
- yp
- c
- a1
- a2
- k1
- k2
- k3
- p1
- p2
- empirical parameter 1
- empirical parameter 2
- empirical parameter 3
- empirical parameter 4
- empirical parameter 5
- empirical parameter 6
- empirical parameter 7
- empirical parameter 8
- empirical parameter 9

*.eop has the following headings in order
- Station ID
- Sensor/Camera ID
- Xo
- Yo
- Zo
- Omega (deg)
- Phi (deg)
- Kappa (deg)

*.xyz has the following headings in order
- Target ID
- X
- Y
- Z
- stdDev of X
- stdDev of Y
- stdDev of Z
