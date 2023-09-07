//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an OpenMP C version of the NPB FT code. This OpenMP  //
//  C version is developed by the Center for Manycore Programming at Seoul //
//  National University and derived from the OpenMP Fortran versions in    //
//  "NPB3.3-OMP" developed by NAS.                                         //
//                                                                         //
//  Permission to use, copy, distribute and modify this software for any   //
//  purpose with or without fee is hereby granted. This software is        //
//  provided "as is" without express or implied warranty.                  //
//                                                                         //
//  Information on NPB 3.3, including the technical report, the original   //
//  specifications, source code, results and information on how to submit  //
//  new results, is available at:                                          //
//                                                                         //
//           http://www.nas.nasa.gov/Software/NPB/                         //
//                                                                         //
//  Send comments or suggestions for this OpenMP C version to              //
//  cmp@aces.snu.ac.kr                                                     //
//                                                                         //
//          Center for Manycore Programming                                //
//          School of Computer Science and Engineering                     //
//          Seoul National University                                      //
//          Seoul 151-744, Korea                                           //
//                                                                         //
//          E-mail:  cmp@aces.snu.ac.kr                                    //
//                                                                         //
//-------------------------------------------------------------------------//

//-------------------------------------------------------------------------//
// Authors: Sangmin Seo, Jungwon Kim, Jun Lee, Jeongho Nah, Gangwon Jo,    //
//          and Jaejin Lee                                                 //
//-------------------------------------------------------------------------//

//---------------------------------------------------------------------
// FT benchmark
//---------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "global.h"
#include "randdp.h"
#include "timers.h"
#include "print_results.h"
#include <NIDAQmx.h>

//---------------------------------------------------------------------
// u0, u1, u2 are the main arrays in the problem.
// Depending on the decomposition, these arrays will have different
// dimensions. To accomodate all possibilities, we allocate them as
// one-dimensional arrays and pass them to subroutines for different
// views
//  - u0 contains the initial (transformed) initial condition
//  - u1 and u2 are working arrays
//  - twiddle contains exponents for the time evolution operator.
//---------------------------------------------------------------------
//---------------------------------------------------------------------
// Large arrays are in common so that they are allocated on the
// heap rather than the stack. This common block is not
// referenced directly anywhere else. Padding is to avoid accidental
// cache problems, since all array sizes are powers of two.
//---------------------------------------------------------------------
/* common /bigarrays/ */
dcomplex u0[NTOTALP];
dcomplex pad1[3];
dcomplex u1[NTOTALP];
dcomplex pad2[3];
//dcomplex u2[NTOTALP];
double twiddle[NTOTALP];
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
static void init_ui(void *ou0, void *ou1, void *ot, int d1, int d2, int d3);
static void compute_initial_conditions(void *ou0, int d1, int d2, int d3);
static double ipow46(double a, int exponent);
static void setup();
static void compute_indexmap(void *ot, int d1, int d2, int d3);
static void fft_init(int n);
static int ilog2(int n);


int main(int argc, char *argv[])
{
  printf("Beginning of simplified main()\n");
  printf("OpenMP process number = %d\n", omp_get_thread_num());
  
  #pragma omp parallel
  printf("Hello from process: %d\n", omp_get_thread_num());

  createTask();
  
  createAIVoltageChan();
  
  setSampleClockAndRate();
 
  startTask();

  takeSamples();
  
  finalize();
  
  printf("End of simplified main V6\n");
  //--------- End of NIDAQ-Test Main() ---------

  int i;
  int iter;
  double total_time, mflops;
  logical verified;
  char Class;

  //---------------------------------------------------------------------
  // Run the entire problem once to make sure all data is touched.
  // This reduces variable startup costs, which is important for such a
  // short benchmark. The other NPB 2 implementations are similar.
  //---------------------------------------------------------------------
  for (i = 1; i <= T_max; i++) {
    timer_clear(i);
  }
  setup();
  init_ui(u0, u1, twiddle, dims[0], dims[1], dims[2]);
  compute_indexmap(twiddle, dims[0], dims[1], dims[2]);
  compute_initial_conditions(u1, dims[0], dims[1], dims[2]);
  fft_init(dims[0]);
  //Adding fft(1, u1, u0) causes PosixError
  return 0;
}


void NIDAQMeasure() {
  printf("Beginning of NIDAQMeasure()\n");
  printf("OpenMP process number = %d\n", omp_get_thread_num());
  printf(">>>> NIDAQMeasure() Called\n");
  printf("I removed createTask, createAIVoltageChan() calls\n");

  printf("NIDAQMeasure()\n");
  createTask();                                                                 
                                                                                
  createAIVoltageChan();                                                        
                                                                                
  setSampleClockAndRate();                                                      
                                                                                
  startTask(); 
                                                                            
  takeSamples();                                                                
                                                                               
  finalize();
  printf("End of NIDAQMeasure()\n");
}

//---------------------------------------------------------------------
// touch all the big data
//---------------------------------------------------------------------
static void init_ui(void *ou0, void *ou1, void *ot, int d1, int d2, int d3)
{
  dcomplex (*u0)[d2][d1+1] = (dcomplex (*)[d2][d1+1])ou0;
  dcomplex (*u1)[d2][d1+1] = (dcomplex (*)[d2][d1+1])ou1;
  double (*twiddle)[d2][d1+1] = (double (*)[d2][d1+1])ot;

  int i, j, k;

  #pragma omp parallel for default(shared) private(i,j,k)
  for (k = 0; k < d3; k++) {
    for (j = 0; j < d2; j++) {
      for (i = 0; i < d1; i++) {
        u0[k][j][i] = dcmplx(0.0, 0.0);
        u1[k][j][i] = dcmplx(0.0, 0.0);
        twiddle[k][j][i] = 0.0;
      }
    }
  }
}

static void compute_initial_conditions(void *ou0, int d1, int d2, int d3)
{
  dcomplex (*u0)[d2][d1+1] = (dcomplex (*)[d2][d1+1])ou0;

  int k, j;
  double x0, start, an, dummy, starts[NZ];

  start = SEED;
  //---------------------------------------------------------------------
  // Jump to the starting element for our first plane.
  //---------------------------------------------------------------------
  an = ipow46(A, 0);
  dummy = randlc(&start, an);
  an = ipow46(A, 2*NX*NY);

  starts[0] = start;
  for (k = 1; k < dims[2]; k++) {
    dummy = randlc(&start, an);
    starts[k] = start;
  }

  //---------------------------------------------------------------------
  // Go through by z planes filling in one square at a time.
  //---------------------------------------------------------------------
  #pragma omp parallel for default(shared) private(k,j,x0)
  for (k = 0; k < dims[2]; k++) {
    x0 = starts[k];
    for (j = 0; j < dims[1]; j++) {
      vranlc(2*NX, &x0, A, (double *)&u0[k][j][0]);
    }
  }
}

//---------------------------------------------------------------------
// compute a^exponent mod 2^46
//---------------------------------------------------------------------
static double ipow46(double a, int exponent)
{
  double result, dummy, q, r;
  int n, n2;

  //---------------------------------------------------------------------
  // Use
  //   a^n = a^(n/2)*a^(n/2) if n even else
  //   a^n = a*a^(n-1)       if n odd
  //---------------------------------------------------------------------
  result = 1;
  if (exponent == 0) return result;
  q = a;
  r = 1;
  n = exponent;

  while (n > 1) {
    n2 = n / 2;
    if (n2 * 2 == n) {
      dummy = randlc(&q, q);
      n = n2;
    } else {
      dummy = randlc(&r, q);
      n = n-1;
    }
  }
  dummy = randlc(&r, q);
  result = r;
  return result;
}

static void setup()
{
  FILE *fp;
  debug = false;

  if ((fp = fopen("timer.flag", "r")) != NULL) {
    timers_enabled = true;
    fclose(fp);
  } else {
    timers_enabled = false;
  }

  niter = NITER_DEFAULT;

  printf("\n\n NAS Parallel Benchmarks (NPB3.3-OMP-C) - FT Benchmark\n\n");
  printf(" Size                : %4dx%4dx%4d\n", NX, NY, NZ);
  printf(" Iterations                  :%7d\n", niter);
  printf(" Number of available threads :%7d\n", omp_get_max_threads());
  printf("\n");

  dims[0] = NX;
  dims[1] = NY;
  dims[2] = NZ;

  //---------------------------------------------------------------------
  // Set up info for blocking of ffts and transposes.  This improves
  // performance on cache-based systems. Blocking involves
  // working on a chunk of the problem at a time, taking chunks
  // along the first, second, or third dimension.
  //
  // - In cffts1 blocking is on 2nd dimension (with fft on 1st dim)
  // - In cffts2/3 blocking is on 1st dimension (with fft on 2nd and 3rd dims)

  // Since 1st dim is always in processor, we'll assume it's long enough
  // (default blocking factor is 16 so min size for 1st dim is 16)
  // The only case we have to worry about is cffts1 in a 2d decomposition.
  // so the blocking factor should not be larger than the 2nd dimension.
  //---------------------------------------------------------------------

  fftblock = FFTBLOCK_DEFAULT;
  fftblockpad = FFTBLOCKPAD_DEFAULT;

  if (fftblock != FFTBLOCK_DEFAULT) fftblockpad = fftblock+3;
}

//---------------------------------------------------------------------
// compute function from local (i,j,k) to ibar^2+jbar^2+kbar^2
// for time evolution exponent.
//---------------------------------------------------------------------
static void compute_indexmap(void *ot, int d1, int d2, int d3)
{
  double (*twiddle)[d2][d1+1] = (double (*)[d2][d1+1])ot;

  int i, j, k, kk, kk2, jj, kj2, ii;
  double ap;

  //---------------------------------------------------------------------
  // basically we want to convert the fortran indices
  //   1 2 3 4 5 6 7 8
  // to
  //   0 1 2 3 -4 -3 -2 -1
  // The following magic formula does the trick:
  // mod(i-1+n/2, n) - n/2
  //---------------------------------------------------------------------

  ap = -4.0 * ALPHA * PI * PI;

  #pragma omp parallel for default(shared) private(i,j,k,kk,kk2,jj,kj2,ii)
  for (k = 0; k < dims[2]; k++) {
    kk = ((k + NZ/2) % NZ) - NZ/2;
    kk2 = kk*kk;
    for (j = 0; j < dims[1]; j++) {
      jj = ((j + NY/2) % NY) - NY/2;
      kj2 = jj*jj + kk2;
      for (i = 0; i < dims[0]; i++) {
        ii = ((i + NX/2) % NX) - NX/2;
        twiddle[k][j][i] = exp(ap * (double)(ii*ii+kj2));
      }
    }
  }
}

//---------------------------------------------------------------------
// compute the roots-of-unity array that will be used for subsequent FFTs.
//---------------------------------------------------------------------
static void fft_init(int n)
{
  int m, nu, ku, i, j, ln;
  double t, ti;

  //---------------------------------------------------------------------
  // Initialize the U array with sines and cosines in a manner that permits
  // stride one access at each FFT iteration.
  //---------------------------------------------------------------------
  nu = n;
  m = ilog2(n);
  u[0] = dcmplx(m, 0.0);
  ku = 2;
  ln = 1;

  for (j = 1; j <= m; j++) {
    t = PI / ln;

    for (i = 0; i <= ln - 1; i++) {
      ti = i * t;
      u[i+ku-1] = dcmplx(cos(ti), sin(ti));
    }

    ku = ku + ln;
    ln = 2 * ln;
  }
}

static int ilog2(int n)
{
  int nn, lg;
  if (n == 1) return 0;
  lg = 1;
  nn = 2;
  while (nn < n) {
    nn = nn*2;
    lg = lg+1;
  }
  return lg;
}