## MatPlotLib Tutorials

"""
Matplotlib is a plotting library for the python programming language and its numerical mathematical extension Numpy
Its provide object orientation API for embedding plot into applications using general-purpose GUI toolkits like Tkinter
wxPython, Qt, or GTK+

Some of the major Props of MatplotLib are:

1. Generally easy to get started for simple plots
2. Support for custom labels and texts
3. Great control of every element in a figure
4. High-quality output in many format
5. Very customization in general
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import lineStyles

# simple Examples

x =np.arange(0,10)
y =np.arange(11,21)

a =np.arange(40,50)
b =np.arange(50,60)

## plotting using matplotlib

##plt scatter

# plt.scatter(x,y,c ="g")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.title("Graph in 2D")
# plt.savefig("Scatter_plot.png") # to save the image
# plt.show()

# plt plot

# y = x * x
# plt.plot(x,y,c ="r", linestyle = "dashed", marker= "*", linewidth= 2, markersize = 12)
# plt.xlabel("X-axis")
# plt.ylabel("y-axis")
# plt.title("2D Diagram")
# plt.savefig("plot_diagram.png")
# plt.grid()
# plt.show()

# create subplot
# no of rows and no of columns

# plt.subplot(2, 2, 1)
# plt.plot(x, y, "r--")  # red dashed line
#
# plt.subplot(2, 2, 2)
# plt.plot(x, y, "g*--")  # green star markers with dashed line
#
# plt.subplot(2, 2, 3)
# plt.plot(x, y, "bo")  # blue circle markers
#
# plt.subplot(2, 2, 4)
# plt.plot(x, y, "s-.y")  # square markers, dash-dot line, blue
#
# plt.savefig("Subplot.png") # save the plot
#
# plt.tight_layout()  # optional: prevents overlap
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.array([0,5,10,15,20,25,30,35,40,45,50])
# x1 = np.array([0,20,35,60,80,90])
#
# # ms = marker size,  mfc = marker face color mec = marker edge color
# plt.plot(x, linestyle = "dashed", color= "green", marker = "*", ms= 6, mfc = "red") # dotted , solid, dashed
# plt.plot(x1,linestyle = "solid", linewidth = 10, color = "orange", marker = ".", ms = 10, mfc= "red", mec= "yellow")
# plt.savefig("plot2.png")
# plt.show()

print(np.pi)

# compute the x and y  coordinates for points on a sine curve

# X = np.arange(0,4 * np.pi, 0.1)
# Y = np.sin(X)
# plt.title("Sine wave form")
#
# # plot the points using matplotlib
# plt.plot(X,Y)
# plt.savefig("sine form.png")
# plt.show()

# Subplot()
# Compute the x and y coordinates for points on sine and cosine curves
# x = np.arange(0, 5 * np.pi, 0.1)
# y_sin = np.sin(x)
# y_cos = np.cos(x)
#
# # Set up a subplot grid that has height 2 and width 1,
# # and set the first such subplot as active.
# plt.subplot(2, 1, 1)
#
# # Make the first plot
# plt.plot(x, y_sin, 'r--')
# plt.title('Sine')
#
# # Set the second subplot as active, and make the second plot.
# plt.subplot(2, 1, 2)
# plt.plot(x, y_cos, 'g-.')
# plt.title('Cosine')
#
# # Show the figure.
# plt.show()

## bar plot

# x = [2,8,10]
# y = [11,6,9]
#
# x1 = [3,9,11]
# y1 = [6,15,7]
#
# plt.bar(x,y, align="center")
# plt.bar(x1,y1, color = "green")
# plt.title("Bar graph")
# plt.xlabel("x-axis")
# plt.ylabel("y-axis")
# plt.savefig("Bar plot")
# plt.show()

## Histograms

# a = np.array([22, 87, 5, 43, 56, 73, 55, 54, 11, 20, 51, 5, 79, 31, 27])
# plt.hist(a, bins=10, color='skyblue', edgecolor='black')  # optional styling
# plt.title("Histogram")
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.savefig("Histplot.png")
# plt.grid(True)
# plt.show()

## Box Plot using Matplotlib

# data = [np.random.normal(0, std, 100) for std in range(1,4)]
#
# # rectangular box plot
# plt.boxplot(data,vert = True, patch_artist=True)
# plt.savefig("boxplot.png")
# plt.show()

# PIE Chart

# Data to plot
labels = 'Python', 'C++', 'Ruby', 'Java'
sizes = [215, 130, 245, 210]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.2, 0, 0, 0)  # explode 1st slice from pie chart

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=False)

plt.savefig("Pie.png")
plt.axis('equal')
plt.show()



