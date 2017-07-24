#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Top Block
# Generated: Mon Jul 24 14:00:31 2017
##################################################

from gnuradio import analog
from gnuradio import blocks
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from optparse import OptionParser
import cuda


class top_block(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Top Block")

        ##################################################
        # Blocks
        ##################################################
        self.cuda_gpu_kernel_0 = cuda.gpu_kernel(0, 1024, 4096, "Auto", "Auto")
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_float*1, 1e6,True)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_float*1, 1024)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_float*1024, "/home/pete/workspace/dwd/gr-cuda/test/data.dat", False)
        self.blocks_file_sink_0.set_unbuffered(False)
        self.analog_const_source_x_0 = analog.sig_source_f(0, analog.GR_CONST_WAVE, 0, 0, 20)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_const_source_x_0, 0), (self.blocks_throttle_0, 0))    
        self.connect((self.blocks_stream_to_vector_0, 0), (self.cuda_gpu_kernel_0, 0))    
        self.connect((self.blocks_throttle_0, 0), (self.blocks_stream_to_vector_0, 0))    
        self.connect((self.cuda_gpu_kernel_0, 0), (self.blocks_file_sink_0, 0))    


def main(top_block_cls=top_block, options=None):

    tb = top_block_cls()
    tb.start()
    try:
        raw_input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
