
NIDAQLIB_DIR="/usr/lib/x86_64-linux-gnu/"
NIDAQLIB_NAME="nidaqmx"

# Already captured by gcc
#INCLUDE_DIR="/usr/include/"

all: countPi calibrate

countPi : countPi.c /home/cobra0/testNIDAQ_Cesar/NPB/SNU_NPB_2019/NPB3.3-OMP-C/FT/ftHelper.o
	gcc -o $@ $^ -L ${NIDAQLIB_DIR} -l ${NIDAQLIB_NAME} -lm

calibrate : calibrate.c
	gcc -o $@ $^ -L ${NIDAQLIB_DIR} -l ${NIDAQLIB_NAME}

.PHONY: clean
clean:
	rm -f countPi	
	rm -f calibrate
