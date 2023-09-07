#!/bin/bash

SAMPLES_PER_SEC=1000
SAMPLING_TIME_IN_SECS=9
START_BENCH_DELAY_SECS=1
OUTPUT_FILE=testout1.csv

echo "Starting Demo! Writing to file: ${OUTPUT_FILE}"

# Start taking samples
#./main $SAMPLES_PER_SEC $SAMPLING_TIME_IN_SECS > ${OUTPUT_FILE} &
./ftHelper3 $SAMPLES_PER_SEC $SAMPLING_TIME_IN_SECS "./Main_OutputV1.txt"
