#include <stdio.h>                                                              
#include <stdlib.h>                                                             
#include <math.h>                                                      
#include <omp.h>                                                             
#include "randdp.h"                                                             
#include "timers.h"


//---- Start of NIDAQ Setup Code From main.c ----

#define DAQmxErrChk(functionCall) if( DAQmxFailed(error=(functionCall)) ) goto Error; else
                                                                                
// This is the name of the DAQ device                                           
#define DAQ_DEVICE_NAME "cDAQ1"                                                 
                                                                                
// This is the name of the module port we measure from                          
#define DAQ_MODULE_NAME "Mod3"                                                  
#define DAQ_DEVICE DAQ_DEVICE_NAME DAQ_MODULE_NAME                              
                                                                                
// The END channel is INCLUSIVE                                                 
// These are the pin indices used for DIFFERENTIAL mode                         
// Pin X will automatically be paired with its corresponding                    
// x+8 pin.                                                                     
// This setup assumes we are using pins 0, 1, 2, and 3                          
// with their complementary pins 8, 9, 10, and 11                               
// which are automatically read by the NIDAQ library                            
// when we are in differential mode                                             
#define DIFF_INPUTS_BEGIN "0"                                                   
#define DIFF_INPUTS_END   "3"                                                   
                                                                                
// We'll pass this into the CreateAIVoltageChan to specify                      
// the channels we want to work with.                                           
#define DIFF_CHANNELS DAQ_DEVICE "/ai" DIFF_INPUTS_BEGIN ":" DIFF_INPUTS_END    
#define PHYS_CHANNELS DIFF_CHANNELS                                             
                                                                                
// The name assigned to our task                                                
#define DAQ_TASK_NAME ""                                                        
                                                                                
// The name we assign to our voltage channel                                    
#define CHANNEL_NAME ""                                                         
                                                                                
// The minimum and maximum volts we expect to measure                           
// The NIDAQ device only supports max of -10/10 volts                           
// This is not good if we have a 12V supply we want to measure                  
#define MIN_VOLTS -10.0                                                         
#define MAX_VOLTS  10.0                                                         
                                                                                
// Number of samples to collect each second for each channel                    
//#define SAMPLES_PER_SEC 1000                                                  
                                                                                
// The number of samples we want to take for each channel                       
//#define SAMPLES_PER_CHANNEL 16000                                             
//#define SAMPLES_PER_CHANNEL 30000                                             
                                                                                
// The amount of time to wait to read the samples                               
#define SAMPLES_WAIT_TIMEOUT_SECS 100                                           
                                                                                
// The number of differential channel pairs we will read from                   
//#define NUM_CHANNEL_PAIRS 4                                                   
#define NUM_CHANNEL_PAIRS 4                                                     
                                                                                
// The number of samples we expect to collect                                   
#define ARRAY_SIZE_IN_SAMPLES NUM_CHANNEL_PAIRS*SAMPLES_PER_CHANNEL             
                                                                                
// The resistance of the resister that we measure the voltage diff over         
#define RESISTOR_OHMS 0.003                                                     
                                                                                
// The voltage we assume that all the lines are running at                      
#define LINE_VOLTAGE 12      

//---- Start of NIDAQ Setup Code From main.c ----



  void init_helper(char* ip_address, int *port) {
    //connect to meter control thread
    printf("---- init ftHelper.c Called ----");
    //pmeter_init(ip_address, port);
    //pmeter_init("127.0.0.1", 80);
  }

  void log_helper(char *log_file, int *option) {
    //Set power profile log file and options
    printf("---- log ftHelper.c Called ----");
    pmeter_log(log_file, option);
  }

  void start_session_helper(char *session_label) {
    //start a new profile session + label it
    printf("---- start_session ftHelper.c Called ----");
    pmeter_start_session(session_label);

  }

  void end_session_helper() {
    //stop current profile session
    printf("---- end_session ftHelper.c Called ----");
    pmeter_end_session();
  }

  void finalize() {
    //Disconnect from the meter control thread
    printf("---- finalize ftHelper.c Called ----");
    pmeter_finalize();

  }



