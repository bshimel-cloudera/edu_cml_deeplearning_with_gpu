########
########  Setup Environment and install libs
########


####
#### Installing Required Libraries
####
!pip install tensorflow --progress-bar off
!pip install scikit-learn pandas --progress-bar off

####
#### The follow line is required to fix an issue with the latest Cuda drivers and tensorflow.
####
!ln -s /usr/local/cuda-11.1/targets/x86_64-linux/lib/libcusolver.so.11.0.1.105 /usr/local/cuda-11.1/targets/x86_64-linux/lib/libcusolver.so.10
