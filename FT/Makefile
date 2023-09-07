SHELL=/bin/sh
BENCHMARK=ft
BENCHMARKU=FT

NIDAQLIB_DIR="/usr/lib/x86_64-linux-gnu/"                                       
NIDAQLIB_NAME="nidaqmx"

include ../config/make.def

include ../sys/make.common

OBJS = ft.o ${COMMON}/${RAND}.o ${COMMON}/print_results.o \
       ${COMMON}/c_timers.o ${COMMON}/wtime.o

${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ftHelper.o ${C_LIB}

.c.o:
	${CCOMPILE} $< ftHelper.o 

ft.o:             ft.c  global.h npbparams.h

clean:
	- rm -f *.o *~ mputil*
	- rm -f ft npbparams.h core
	- if [ -d rii_files ]; then rm -r rii_files; fi
