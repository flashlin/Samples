import numpy as np
import matplotlib.pyplot as plt

# Create a set of x values from 0 to 4*pi
x = np.linspace(0, 4*np.pi, 1000)

# Create a set of y values using the sin function
y = np.sin(x)

# Create the plot
plt.figure(figsize=(5, 8))
plt.plot(y, x)
plt.title('Waveform 2D Plot')
plt.ylabel('Y')
plt.xlabel('X')
plt.grid(True)
plt.show()
