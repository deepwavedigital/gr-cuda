title: gr-cuda
brief: Library that integrates CUDA GPU processing with GNU Radio
tags: GPU,  CUDA
author: [Peter Witkowski](https://github.com/dwd-pete)
copyright_owner: Deepwave Digital, Inc. <https://deepwavedigital.com>

dependencies:

  - gnuradio (v3.7.9 or newer)
  - pycuda (v2017.1 or newer)
---

This OOT module contains experimental code on integration of GPU processing into
GNU Radio by using the PyCUDA library to run CUDA code from within GNU Radio.

# Current Blocks
-> gpu_kernel.py: This is a Python block that runs an arbitrary CUDA kernel from
                  within GNU Radio. Currently the CUDA kernel simply divides
                  every sample by two, but this can be changed by the end user
                  if desired.

# Building
1. Install Dependencies
2. Clone the gr-cuda repo
3. Install the OOT Module
    -> cd gr-cuda; mkdir build; cd build
    -> cmake ../
    -> make
    -> sudo make install

# Uninstall Block from GNU Radio Companion
1. cd gr-cuda/build
2. sudo make uninstall

