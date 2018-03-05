from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six


class Hparams(object):
  def __init__(self, **kwargs):
    print("-" * 80)
    print("Creating a Hparams object")

    for name, value in six.iteritems(kwargs):
      self.add_param(name, value)

  def add_param(self, name, value):
    setattr(self, name, value)


class Iwslt16EnDeBpe32SharedParams(Hparams):
  """Basic params for the data set. Set other hparams via inheritance."""


  def __init__(self, **kwargs):
    super(Iwslt16EnDeBpe32SharedParams, self).__init__(**kwargs)
    self.dataset = "IWSLT 2016 En-De with BPE 32K Shared Vocab"

    self.unk = "<unk>"
    self.bos = "<s>"
    self.eos = "</s>"
    self.pad = "<pad>"

    self.unk_id = None
    self.eos_id = None
    self.bos_id = None
    self.pad_id = None

