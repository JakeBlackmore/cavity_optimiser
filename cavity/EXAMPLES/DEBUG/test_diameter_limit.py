import numpy
import scipy.optimize as optimize
import core
import matplotlib.pyplot as plt
Vmax = 12e-12
R = 250e-6

print(core.spherical_cap(0.0001414,4.013e-5))
fn = lambda x: Vmax-core.spherical_cap(R,x)[0]


print(fn(2e-6),fn(100e-6),fn(2*R))

root = optimize.root_scalar(fn,bracket=[2e-6,2*R])
x = root.root
print(x)

d = numpy.linspace(0,2*R,5000)

fn_d = [fn(i) for i in d]

plt.plot(d,fn_d)
plt.show()
