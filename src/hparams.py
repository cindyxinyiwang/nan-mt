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

    self.vocab_size = 28000 + 1
    self.unk = "<unk>"
    self.bos = "<s>"
    self.eos = "</s>"
    self.unk_id = 27997
    self.eos_id = 27998
    self.bos_id = 27999

    self.pad = "<pad>"
    self.pad_id = 28000

