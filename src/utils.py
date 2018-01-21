from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

from datetime import datetime

class Logger(object):
  def __init__(self, output_file):
    self.terminal = sys.stdout
    self.log = open(output_file, "a")

  def write(self, message):
    self.terminal.write(message)
    self.terminal.flush()
    self.log.write(message)
    self.log.flush()

