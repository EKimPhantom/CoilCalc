from CoilCalc import Coil, Field, analyze_signal
import numpy as np

from matplotlib import pyplot as plt

field = Field([[1, 5, 0], [2, 20, 0], [3, 10, 0]])
coil = Coil(10, -10, 25, 50, 2 * np.pi/10)
t = np.arange(0,20,0.05)
length = np.arange(0,coil.L,0.05)
straight = coil.straight_coil_signal(field, t)
sine_deform = coil.curved_coil_signal(field, t, coil.sine_deformation)

gradients1, bin_axis1 = analyze_signal(straight, t)
gradients2, bin_axis2 = analyze_signal(sine_deform, t)

plot_width = 0.2
fig, axs = plt.subplots(2,1)
line1, = axs[0].plot(t, straight.signal, color='r', label='Straight Coil')
line2, = axs[0].plot(t, sine_deform.signal, color='b', label='Sine Deformation Coil')
axs[0].legend(handles=[line1, line2], loc='upper right')
axs[0].set_title("Signal")

axs[1].bar(bin_axis1 - plot_width*2/3, gradients1, color='r', width=plot_width, label='Straight Coil')
axs[1].bar(bin_axis2 + plot_width*2/3, gradients2, color='b', width=plot_width, label='Sine Deformation Coil')
axs[1].set_title("Spectrum")
axs[1].set_xlim((0, 10))
axs[1].set_xticks(np.arange(0,10))

plt.tight_layout()
plt.show()

plt.plot(length, coil.r1 * np.ones_like(length), length, coil.r2 * np.ones_like(length), color='r')
plt.plot(length, coil.sine_deformation(length) + coil.r1 * np.ones_like(length), length, coil.sine_deformation(length) + coil.r2 * np.ones_like(length), color='b')
plt.show()