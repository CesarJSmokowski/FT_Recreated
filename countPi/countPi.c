#include <stdlib.h>                                                             
#include <stdio.h>                                                              
#include <NIDAQmx.h>                                                            
                                                                                
//Includes for writing to temp files                                            
#include<unistd.h>                                                              
#include<string.h>                                                              
#include<errno.h>
#include<math.h>

int main()
//int main(int argc, char *argv[])
{
 long int i, n;
 double sum=0.0, term, pi;
 
 printf("Enter number of terms: ");
 scanf("%ld", &n);
 //n = atoi(argv[1]);

  createTask();                                                                 
  createAIVoltageChan();                                                        
  createAIVoltageChan();                                                        
  setSampleClockAndRate();                                                      
  startTask();                                                                  
  takeSamples();                                                                

 /* Applying Leibniz Formula */
 for(i=0;i< n;i++)
 {
  term = pow(-1, i) / (2*i+1);
  sum += term;
 }
 pi = 4 * sum;

  finalize();

 printf("\nPI = %.6lf", pi);
 
 return 0;
}
