#!/bin/bash


CURRENT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_PATH="../../libs/KLab/main/unix/"


cd $BUILD_PATH
. UnixMakefile_x64Debug_Make.sh

cd $CURRENT_PATH
cd $BUILD_PATH
cd UnixMakefile_x64Debug
. build.sh

cd $CURRENT_PATH
cd $BUILD_PATH
. UnixMakefile_x64Release_Make.sh

cd $CURRENT_PATH
cd $BUILD_PATH
cd UnixMakefile_x64Release
. build.sh
