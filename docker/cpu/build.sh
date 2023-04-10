#!/bin/bash


start=`date +%s`
if [ $1 == "arm" ]
then
	nohup docker build -t neurorobotics arm/ > build.log &
else
	nohup docker build -t neurorobotics x86_64/ > build.log &
fi
end=`date +%s`
runtime=$((end-start))
echo "Build Runtime:" > build.log
echo $runtime > build.log
echo "Done. Thank you."
