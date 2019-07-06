import sys
import os
for line in sys.stdin:
    line = line.strip()
    group = line.split(";")
    out_s = " ".join([str(len(x.split())) for x in group])
    print(out_s)
    
