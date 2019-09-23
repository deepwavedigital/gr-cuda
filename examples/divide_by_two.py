#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: AIR-T Example
# Author: Deepwave Digital
# Generated: Mon Sep 23 18:55:03 2019
##################################################

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print "Warning: failed to XInitThreads()"

from PyQt4 import Qt
from gnuradio import blocks
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio import qtgui
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from optparse import OptionParser
import cuda
import numpy
import sip
import soapy
import sys


class divide_by_two(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "AIR-T Example")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("AIR-T Example")
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "divide_by_two")
        self.restoreGeometry(self.settings.value("geometry").toByteArray())

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 31.25e6
        self.frequency = frequency = 2.4e9
        self.buffer_size = buffer_size = 16384

        ##################################################
        # Blocks
        ##################################################
        self.soapy_source_0 = \
          soapy.source(1, "device=SoapyAIRT", samp_rate, "fc32")
        
        
        
        
        self.soapy_source_0.set_frequency(0, frequency)
        self.soapy_source_0.set_gain(0, 0)
        self.soapy_source_0.set_gain_mode(0, True)
        self.soapy_source_0.set_dc_offset_mode(0, True)
         
        self.qtgui_time_sink_x_0 = qtgui.time_sink_c(
        	512, #size
        	samp_rate, #samp_rate
        	"", #name
        	2 #number of inputs
        )
        self.qtgui_time_sink_x_0.set_update_time(0.10)
        self.qtgui_time_sink_x_0.set_y_axis(-0.11, 0.11)
        
        self.qtgui_time_sink_x_0.set_y_label("Signal Amplitude", "")
        
        self.qtgui_time_sink_x_0.enable_tags(-1, False)
        self.qtgui_time_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_NORM, qtgui.TRIG_SLOPE_POS, 0, 0, 0, "")
        self.qtgui_time_sink_x_0.enable_autoscale(False)
        self.qtgui_time_sink_x_0.enable_grid(True)
        self.qtgui_time_sink_x_0.enable_control_panel(False)
        
        if not True:
          self.qtgui_time_sink_x_0.disable_legend()
        
        labels = ["Input Signal (Real)", "Input Signal (Imag)", "Output Signal (Real)", "Output Signal (Imag)", "",
                  "", "", "", "", ""]
        widths = [2, 2, 2, 2, 1,
                  1, 1, 1, 1, 1]
        colors = ["black", "green", "Dark Blue", "magenta", "cyan",
                  "magenta", "yellow", "dark red", "dark green", "blue"]
        styles = [2, 2, 1, 1, 1,
                  1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
                   -1, -1, -1, -1, -1]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0]
        
        for i in xrange(2*2):
            if len(labels[i]) == 0:
                if(i % 2 == 0):
                    self.qtgui_time_sink_x_0.set_line_label(i, "Re{{Data {0}}}".format(i/2))
                else:
                    self.qtgui_time_sink_x_0.set_line_label(i, "Im{{Data {0}}}".format(i/2))
            else:
                self.qtgui_time_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_0.set_line_alpha(i, alphas[i])
        
        self._qtgui_time_sink_x_0_win = sip.wrapinstance(self.qtgui_time_sink_x_0.pyqwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_0_win)
        self.cuda_gpu_kernel_0 = cuda.gpu_kernel(0, numpy.complex64, buffer_size, 128)
        self.blocks_vector_to_stream_0 = blocks.vector_to_stream(gr.sizeof_gr_complex*1, buffer_size)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, buffer_size)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_stream_to_vector_0, 0), (self.cuda_gpu_kernel_0, 0))    
        self.connect((self.blocks_vector_to_stream_0, 0), (self.qtgui_time_sink_x_0, 1))    
        self.connect((self.cuda_gpu_kernel_0, 0), (self.blocks_vector_to_stream_0, 0))    
        self.connect((self.soapy_source_0, 0), (self.blocks_stream_to_vector_0, 0))    
        self.connect((self.soapy_source_0, 0), (self.qtgui_time_sink_x_0, 0))    

    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "divide_by_two")
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()


    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.qtgui_time_sink_x_0.set_samp_rate(self.samp_rate)

    def get_frequency(self):
        return self.frequency

    def set_frequency(self, frequency):
        self.frequency = frequency
        self.soapy_source_0.set_frequency(0, self.frequency)

    def get_buffer_size(self):
        return self.buffer_size

    def set_buffer_size(self, buffer_size):
        self.buffer_size = buffer_size


def main(top_block_cls=divide_by_two, options=None):

    from distutils.version import StrictVersion
    if StrictVersion(Qt.qVersion()) >= StrictVersion("4.5.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()
    tb.start()
    tb.show()

    def quitting():
        tb.stop()
        tb.wait()
    qapp.connect(qapp, Qt.SIGNAL("aboutToQuit()"), quitting)
    qapp.exec_()


if __name__ == '__main__':
    main()
