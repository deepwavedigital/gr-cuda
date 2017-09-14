#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Copyright 2017 Deepwave Digital Inc.
# 
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
# 
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
# 

from gnuradio import gr

import numpy
import pycuda.compiler
import pycuda.driver

class gpu_kernel(gr.sync_block):
  """Block that executes a CUDA kernel found within its Python source code. Built from a Sync Block since the kernel in this case has a 1:1 relationship between input and output.  All CUDA resources (i.e., device context, compiled kernel code, pointers to device memory, etc.) are managed within this block.

    Args:
      device_num: CUDA device number (0 if only one GPU installed). You can verify the device number by running the CUDA utility "deviceQuery".
      io_type: Data type to perform processing on. Since the kernel takes in floats, this value can either be "Float" for 32-bit floating point samples or "Complex", which are two 32-bit floating point samples back-to-back (one representing the real component and the other representing the imaginary component).
      vlen: Length of the input vector. In the case of our CUDA kernel, this is how many samples we process for every kernel invocation.
      threads_per_block: # of threads per CUDA block. Defaults to 128, since (in theory) this value should be optimal for the Jetson TX2. Note that our kernel only operates in one dimension and that grid dimensions are automatically computed based on this value.
  """
  def __init__(self, device_num, io_type, vlen, threads_per_block):
    gr.sync_block.__init__(self,
      name="gpu_kernel",
      in_sig=[(io_type, vlen)],
      out_sig=[(io_type, vlen)])
    # Initialize PyCUDA stuff...
    pycuda.driver.init()
    device = pycuda.driver.Device(device_num)
    context_flags = \
      (pycuda.driver.ctx_flags.SCHED_AUTO | pycuda.driver.ctx_flags.MAP_HOST)
    self.context = device.make_context(context_flags)
    # Build the kernel here.  Alternatively, we could have compiled the kernel
    # beforehand with nvcc, and simply passed in a path to the executable code.
    compiled_cuda = pycuda.compiler.compile("""
      // Simple kernel that takes every input and divides by two
      __global__ void divide_by_two(const float* const in,
                                    float* const out,
                                    const size_t num_floats) {
        static const float kDivideByMe = 2.0;
        const int i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i < num_floats) {
          out[i] = in[i] / kDivideByMe;
        }
      }
    """)
    module = pycuda.driver.module_from_buffer(compiled_cuda)
    self.kernel = module.get_function("divide_by_two").prepare(["P", "P", "q"])
    self.threads_per_block = threads_per_block
    # Allocate device mapped pinned memory
    self.sample_type = io_type
    self.sample_size = numpy.dtype(self.sample_type).itemsize
    self.mapped_host_malloc(vlen)
    self.context.pop()

  def mapped_host_malloc(self, num_samples):
    self.mapped_host_input = \
      pycuda.driver.pagelocked_zeros(
        num_samples,
        self.sample_type,
        mem_flags = pycuda.driver.host_alloc_flags.DEVICEMAP)
    self.mapped_host_output = \
      pycuda.driver.pagelocked_zeros(
        num_samples,
        self.sample_type,
        mem_flags = pycuda.driver.host_alloc_flags.DEVICEMAP)
    self.mapped_gpu_input = self.mapped_host_input.base.get_device_pointer()
    self.mapped_gpu_output = self.mapped_host_output.base.get_device_pointer()
    self.num_samples = num_samples;
    self.num_floats = self.num_samples;
    if (self.sample_type == numpy.complex64):
      # If we're processing complex data, we have two floats for every sample...
      self.num_floats *= 2
    self.num_blocks = self.num_floats / self.threads_per_block
    left_over_samples = self.num_floats % self.threads_per_block
    if (left_over_samples != 0):
      # If vector length is not an even multiple of the number of threads in a
      # block, we need to add another block to process the "leftover" samples.
      self.num_blocks += 1

  def mapped_host_realloc(self, num_samples):
    del self.mapped_host_input
    del self.mapped_host_output
    self.mapped_host_malloc(num_samples)

  def work(self, input_items, output_items):
    in0 = input_items[0]
    out = output_items[0]
    self.context.push()
    recv_samples = in0.shape[1]
    if (recv_samples > self.num_samples):
      print "Warning: Not Enough GPU Memory Allocated. Reallocating..."
      print "-> Required Space: %d Samples" % recv_samples
      print "-> Allocated Space: %d Samples" % self.num_samples
      self.mapped_host_realloc(recv_samples)
    # Launch a kernel for each vector we received. If the vector is large enough
    # we should only be receiving one or two vectors for every time the GNU
    # Radio scheduler calls work().
    for i in range(0, in0.shape[0]):
      self.mapped_host_input[0:in0.shape[1]] = in0[i, 0:in0.shape[1]]
      self.kernel.prepared_call((self.num_blocks, 1, 1),
                                (self.threads_per_block, 1, 1),
                                self.mapped_gpu_input,
                                self.mapped_gpu_output,
                                self.num_floats)
      self.context.synchronize()
      out[i, 0:in0.shape[1]] = self.mapped_host_output[0:in0.shape[1]]
    self.context.pop()
    return len(output_items[0])

