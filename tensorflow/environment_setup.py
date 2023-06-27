########
########  Setup Environment and install libs
########

# This version tested with Runtime:
#  Editor: Workbench
#  Kernel: Python 3.7
#  Edition: Nvidia GPU
#  Version: 2023.05f

####
#### Installing Required Libraries
####
!pip install tensorflow==2.9.1 --progress-bar off
!pip install scikit-learn==1.0.2 pandas==1.3.5 --progress-bar off

####
#### The follow line is required to fix an issue with the latest Cuda drivers and tensorflow.
####
!ln -s /usr/local/cuda-11.4/targets/x86_64-linux/lib/libcusolver.so.11.2.0.120 /usr/local/cuda-11.4/targets/x86_64-linux/lib/libcusolver.so.10
!echo export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.4/compat:/usr/local/cuda-11.4/lib64 >>.bash_profile
