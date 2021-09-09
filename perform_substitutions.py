import glob
def file_get_contents(filename):
    with open(filename) as f:
        return f.read()

files = glob.glob("substitutions/*")
replacements = {}
for fname in files:
    replacements[fname.replace("substitutions/","").split(".")[0]]=file_get_contents(fname)
    
    

import sys

for line in sys.stdin:
    for key in replacements:
        line = line.replace("&&&"+key+"&&&",replacements[key])
    print line
    