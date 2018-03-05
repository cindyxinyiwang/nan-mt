
import sys

file_name = sys.argv[1]
with open(file_name, 'r') as myfile:
  for line in myfile:
    toks = line.split()
    print(' '.join(toks[:-1]))
