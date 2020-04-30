#!/usr/bin/env python3

j = 5
# a is rad along length
a = 0.001
# b is rad across width
b = 0.0001

f = open("ParametricFile_Test.py","w+")

f.write("CalcName = [")
for i in range(j-1):
    f.write("'Parametric%d'," % (i+1))
f.write("'Parametric%d'" % (j))
f.write("]\n")

f.write("MeshName = [")
for i in range(j-1):
    f.write("'Notch%d'," % (i+1))
f.write("'Notch%d'" % (j))
f.write("]\n")

f.write("Rad_a = [")
for i in range(j-1):
    f.write("%f," % (a*(i+1)))
f.write("%f" % (a*j))
f.write("]\n")

f.write("Rad_b = [")
for i in range(j-1):
    if i==0:
        f.write("%f," % (b))
    else:
        f.write("%f," % (b*5*(i)))
f.write("%f" % (b*5*(j-1)))
f.write("]\n")

f.close()

