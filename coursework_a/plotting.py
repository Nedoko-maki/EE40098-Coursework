# Import the matplotlib pyplot library
# It has a very long name, so import it as the name plt
import matplotlib.pyplot as plt
# Make a new plot (XKCD style)
fig = plt.xkcd()
# Add points as scatters - scatter(x, y, size, color)
plt.plot([-0.5, 1], [1, -0.5], color="blue")
plt.plot([-0.5, 2], [2, -0.5], color="blue")
# zorder determines the drawing order, set to 3 to make the
# grid lines appear behind the scatter points
plt.scatter(0, 0, s=50, color="red", zorder=3)
plt.scatter(0, 1, s=50, color="green", zorder=3)
plt.scatter(1, 0, s=50, color="green", zorder=3)
plt.scatter(1, 1, s=50, color="red", zorder=3)
# Set the axis limits
plt.xlim(-0.5, 2)
plt.ylim(-0.5, 2)
# Label the plot
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.title("State Space of Input Vector (XOR)")
#plt.annotate('', xy=(1, 0.5), xytext=(1.5, 1.5),
#           arrowprops=dict(facecolor='black', shrink=0.05))
# Turn on grid lines
plt.grid(True, linewidth=1, linestyle=":")
# Autosize (stops the labels getting cut off)
plt.tight_layout()
# Show the plot
plt.show()

