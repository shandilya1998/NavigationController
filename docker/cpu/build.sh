#!/bin/bash

start=`date +%s`
nohup docker build -t neurorobotics . > build.log &
end=`date +%s`
runtime=$((end-start))
echo "Build Runtime:" > build.log
echo $(runtime) > build.log
echo "Done. Thank you."
