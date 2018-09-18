# List of things to do
This list is in no particular order of importance

## Break down massive F array
In the WiDIM function, there is a massive array that holds all the variables for every iteration.
It is insanely hard to work with for the GPU, and should be broken down into multiple smaller arrays. 

## Multithreading
Spin off multiple threads to utilize multiple GPUs

## Port to Python 3
I don't think this would be that hard. Change up some print statements, get rid of the progress bar

## Move GPU functions to new file
Move some GPU functions to a new file. Makes the code cleaner and makes development/testing easier.

## Check dependencies
Need to auto install dependencies, or at least look for dependencies and then list the ones
that are not on the system.



