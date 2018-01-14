import sys
import matplotlib.pyplot as plt

for line in open(sys.argv[1]):
    a = line.split(" ")
for line in open(sys.argv[2]):
    b = line.split(" ")
for line in open(sys.argv[3]):
    c = line.split(" ")
for line in open(sys.argv[4]):
    d = line.split(" ")

a.insert(0,0)
b.insert(0,0)
c.insert(0,0)
d.insert(0,0)

plt.figure()
plt.title("dev accuracy per number of lines trained")
plt.xlabel("number of lines [x100]")
plt.ylabel("accuracy [%] ")
plt.plot(map(lambda n: n * 5, range(len(a))), map(float, a) , 'C0', label='a')
plt.plot(map(lambda n: n * 5, range(len(b))), map(float, b) , 'C1', label='b')
plt.plot(map(lambda n: n * 5, range(len(c))), map(float, c) , 'C2', label='c')
plt.plot(map(lambda n: n * 5, range(len(d))), map(float, d) , 'C3', label='d')
plt.legend()
plt.savefig("graph1.png")
