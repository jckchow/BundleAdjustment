# BundleAdjustment

Date: December 21, 2017

1. Run cmake with destination set to a new folder you create called "build"
2. Navigate to the build folder and run "make" to compile
3. Call ./bundleAdjustment to run the code

Note: If the y-axis is set to -1 it means we have a LHS and currently the code doesn't do anything with it. The user need to manually do the following 3 things:

- Change image observation y to -y
- Change yp to -yp
- Change p2 to -p2 (tangential component of the DLD)
