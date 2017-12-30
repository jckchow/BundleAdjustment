###
# Author: Jacky Chow
# -DCMAKE_BUILD_TYPE=Release
###


#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++98")

cmake_minimum_required(VERSION 2.8)
project(bundleAdjustment)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

#find_package(PCL 1.8 REQUIRED COMPONENTS common io)
#include_directories(${PCL_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})

#find_package(Boost 1.65)
#include_directories(${Boost_INCLUDE_DIRS})
#link_directories(${Boost_LIBRARY_DIRS})

# targetProject name of project, should be same as name of folder where this cmakefile exists. 
function(createTarget targetProject)	
	file(GLOB _CPP "src/*.cpp")
	file(GLOB _H "include/*.h")

	source_group("Header Files" FILES ${_H})
	source_group("Source Files" FILES ${_CPP})

    add_executable(${targetProject} ${_H} ${_CPP})
    
endfunction()

# Bundle Adjustment
createTarget(bundleAdjustment)
#target_link_libraries(bundleAdjustment ${CERES_LIBRARIES} ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} ${PCL_LIBRARIES} )
#target_link_libraries(bundleAdjustment ${CERES_LIBRARIES} ${PYTHON_LIBRARIES} ${PCL_LIBRARIES} )
target_link_libraries(bundleAdjustment ${CERES_LIBRARIES} ${PYTHON_LIBRARIES} )
