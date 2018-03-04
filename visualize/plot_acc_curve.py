"""Code to read a stdout file and plot an attribute.""" 

import os
import sys
import time
import numpy as np
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["lines.linewidth"] = 1.0
mpl.rcParams["grid.color"] = "k"
mpl.rcParams["grid.linestyle"] = ":"
mpl.rcParams["grid.linewidth"] = 0.5
mpl.rcParams["font.size"] = 12
mpl.rcParams["legend.fontsize"] = "large"
mpl.rcParams["legend.framealpha"] = None
mpl.rcParams["figure.titlesize"] = "medium"

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", None, "")
flags.DEFINE_string("folder_names", None, "")
flags.DEFINE_string("plot_name", None, "")
flags.DEFINE_boolean("plot_valid", True, "Plot valid accuracy or not")

def plot_result():
  xmin, xmax = 1e9, -1e9
  ymin, ymax = 1e9, -1e9

  fig = plt.figure()

  folder_names = FLAGS.folder_names.split(",")
  colors = [
    (1.0, 0.0, 0.0, 1.0),
    (0.0, 0.0, 1.0, 1.0),
    (0.0, 0.5, 0.0, 1.0),
    (1.0, 0.0, 1.0, 1.0),
    (1.0, 0.0, 0.5, 1.0),
  ]
  for folder_name, color in zip(folder_names, colors):
    full_name = os.path.join(FLAGS.output_dir, folder_name, "stdout")

    valid_acc, test_acc = [], []

    with open(full_name, "r") as finp:
      lines = finp.read().split("\n")

    for line in lines:
      line = line.strip()
      if line.startswith("valid_accuracy"):
        valid_acc.append(float(line.split(" ")[-1]))
      elif line.startswith("test_accuracy"):
        test_acc.append(float(line.split(" ")[-1]))

    assert len(valid_acc) == len(test_acc), "different recorded valid and test"
    steps = np.arange(0, len(valid_acc)) + 1

    xmin = min(xmin, np.min(steps))
    xmax = max(xmax, np.max(steps))
    ymin = 0.75  # min(ymin, min(np.min(valid_acc), np.min(test_acc)))
    ymax = 1.00  # max(ymax, max(np.max(valid_acc), np.max(test_acc)))

    if FLAGS.plot_valid:
      plt.plot(steps, valid_acc, color=color, linestyle=":",
               label="{}_valid".format(folder_name))
      plt.plot(steps, test_acc, color=color, linestyle="-",
               label="{}_test".format(folder_name))
    else:
      plt.plot(steps, test_acc, color=color, linestyle="-",
               label="{}".format(folder_name))

  plt.gca().set_xlabel("Epoch")
  plt.gca().set_xlim(xmin=xmin, xmax=xmax)
  plt.gca().set_ylim(ymin=ymin, ymax=ymax)
  plt.gca().set_ylabel("Accuracy")
  plt.gca().set_title(FLAGS.plot_name)
  plt.gca().grid(True)
  plt.legend(loc="best", fontsize=8)
  plt.tight_layout()
  
  plt.savefig("{}.png".format(FLAGS.plot_name))


def main(_):
  assert FLAGS.output_dir is not None, "please specify --output_dir"
  assert FLAGS.folder_names is not None, "please specify --folder_names"
  assert FLAGS.plot_name is not None, "please specify --plot_name"

  plot_result()


if __name__ == "__main__":
  tf.app.run()
