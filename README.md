# GR-CUDA
Incorporate GPU Programming into GNU Radio

### Author
<p align="center">
<img src="https://deepwavedigital.com/media/images/dwd2_crop_transparent.png" Width="50%" />
</p>

This software is written by **Deepwave Digital, Inc.** [www.deepwavedigital.com]().

&nbsp;
***
## Description
This OOT module contains experimental code on integration of GPU processing into
GNU Radio by using the PyCUDA library to run CUDA code from within GNU Radio.
For a detailed tutorial on using this module see this tutorial:
[http://docs.deepwavedigital.com/Tutorials/4_gr-cuda.html](http://docs.deepwavedigital.com/Tutorials/4_gr-cuda.html).

<p align="center">
<img src="https://deepwavedigital.com/wp-content/uploads/2019/09/gr-cuda-edited.gif" Width="70%" />
</p>

## Current Blocks
**gpu_kernel.py** - This is a Python block that runs an arbitrary CUDA kernel
                    from within GNU Radio. Currently the CUDA kernel simply
                    divides every sample by two, but this can be changed by the
                    end user if desired.

## Building the Tutorial

### Install Dependencies

#### CUDA Dependencies
Depending on the version of AirStack you have, the installation procedure may
slightly differ. This tutorial is written to be compatible with the AIR-T build.
Any usage for non-AIR-T devices is left to the user.

##### Airstack Version 0.1
This version includes a copy of GNU Radio installed for all users. In order to
use gr-cuda in this environment, we need to install the PyCUDA package for
Python 2.7 for all users. Use the below procedure to complete this installation,
which will place the installed copy of PyCUDA into
`/usr/local/lib/python2.7/site-packages`.
```
$ sudo -i
$ PATH=/usr/local/cuda-9.0/bin:${PATH} pip install pycuda
```
and then ctrl-d to get back to a non-root shell.

#### Other Dependencies
Other than the CUDA specific dependencies, the dependencies are the same as
building any other OOT module. See the tutorial here from GNU Radio:
[https://wiki.gnuradio.org/index.php/Guided_Tutorial_GNU_Radio_in_Python](https://wiki.gnuradio.org/index.php/Guided_Tutorial_GNU_Radio_in_Python)

### Download GR-CUDA Tutorial
Clone the `gr-cuda` repo. We recommend installing it in the same location as
gr-wavelearner.
```
$ cd /usr/local/src/deepwave
$ git clone https://github.com/deepwavedigital/gr-cuda.git
```

### Install Package
To install the OOT Module, follow these steps:

```
$ cd gr-cuda
$ mkdir build
$ cd build
$ cmake ../
$ make
$ sudo make install
```

## Uninstall
If you would level like to uninstall the GR-CUDA tutorial:
```
$ cd /usr/local/src/deepwave/gr-cuda/build
$ sudo make uninstall
```

&nbsp;
***
### Tags
CUDA, pyCUDA, GPGPU, GNU Radio, SDR, GPU Programming, Jetson, AIR-T, NVIDIA,
Deepwave Digital

### Credits and License
GR-CUDA is designed and written by **Deepwave Digital, Inc.**
[www.deepwavedigital.com]() and is licensed under the GNU General Public
License. Copyright notices at the top of source files.
