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
  markers = ["v", "x", "^", ">", "o"]
  for folder_name, color, marker in zip(folder_names, colors, markers):
    full_name = os.path.join(FLAGS.output_dir, folder_name, "stdout")

    valid_acc, valid_ppl = {}, {}

    with open(full_name, "r") as finp:
      lines = finp.read().split("\n")

    for line in lines:
      line = line.strip()
      if not line:
        continue
      if line.startswith("val_step"):
        tokens = line.split(" ")
        step, acc, ppl = None, None, None
        for token in tokens:
          token = token.strip()
          if not token:
            continue
          if token.startswith("val_step="):
            step = int(token.split("=")[-1]) // 1000
          elif token.startswith("acc="):
            acc = float(token.split("=")[-1])
          elif token.startswith("val_ppl="):
            ppl = float(token.split("=")[-1])
        if (step is not None) and (acc is not None) and (ppl is not None):
          valid_acc[step] = acc
          valid_ppl[step] = ppl

    steps, accs, ppls = [], [], []
    for step, acc in sorted(valid_acc.iteritems()):
      steps.append(step)
      accs.append(acc)
    for _, ppl in sorted(valid_ppl.iteritems()):
      ppls.append(ppl)

    steps = np.array(steps, dtype=np.int32)
    xmin = min(xmin, np.min(steps))
    xmax = max(xmax, np.max(steps))
    if "acc" in FLAGS.plot_name:
      plt.plot(steps, accs, color=color, linestyle="-", marker=marker,
               label="{}_valid".format(folder_name))
    else:
      assert "ppl" in FLAGS.plot_name, "What do you want to plot?"
      plt.plot(steps, np.log(ppls), color=color, linestyle="-", marker=marker,
               label="{}_valid".format(folder_name))

  if "ppl" in FLAGS.plot_name:
    xmin = 0.0
    xmax = 120.0
    ymin = 2.42  # min(ymin, min(np.min(valid_acc), np.min(test_acc)))
    ymax = 4.20  # max(ymax, max(np.max(valid_acc), np.max(test_acc)))
    dist = (ymax - ymin) / 10
    plt.yticks(np.arange(ymin, ymax + dist, dist))
  else:
    xmin = 0.0
    xmax = 120.0
    ymin = 0.30  # min(ymin, min(np.min(valid_acc), np.min(test_acc)))
    ymax = 0.60  # max(ymax, max(np.max(valid_acc), np.max(test_acc)))

  dist = (xmax - xmin) / 10
  plt.xticks(np.arange(0, xmax + dist, dist))
  plt.gca().set_xlabel("Step (x1000)")
  plt.gca().set_xlim(xmin=xmin, xmax=xmax)
  plt.gca().set_ylim(ymin=ymin, ymax=ymax)
  plt.gca().set_ylabel("log ppl" if "ppl" in FLAGS.plot_name else "acc")
  plt.gca().set_title(FLAGS.plot_name)
  plt.gca().grid(True)
  plt.legend(loc="lower right" if "acc" in FLAGS.plot_name else "upper right",
             fontsize=8)
  plt.tight_layout()
  
  plt.savefig("{}.png".format(FLAGS.plot_name))


def main(_):
  assert FLAGS.output_dir is not None, "please specify --output_dir"
  assert FLAGS.folder_names is not None, "please specify --folder_names"
  assert FLAGS.plot_name is not None, "please specify --plot_name"

  plot_result()


if __name__ == "__main__":
  tf.app.run()
