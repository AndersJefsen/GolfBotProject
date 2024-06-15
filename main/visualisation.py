import matplotlib.pyplot as plt
import numpy as np
import threading
import queue
import time


class Visualisation():
    def __init__(self):
        self.array = np.ones((121, 166))
        self.queue = queue.Queue()
        self.points = []  # List to keep track of points and their labels
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.im = self.ax.imshow(self.array, cmap='Greys', interpolation='none', origin='lower')
        self.ax.axis('on')  # Turn off the axis


    def update_plot(self, frame):
        while not self.queue.empty():
            x, y, name = self.queue.get()
            self.points.append((x, y, name))
            self.array[y, x] = 0  # Mark the point on the array
        self.ax.clear()
        self.im = self.ax.imshow(self.array, cmap='Greys', interpolation='none', origin='lower')
        self.ax.axis('on')  # Turn off the axis
        for px, py, pname in self.points:
            self.ax.text(px, py, pname, color='red', fontsize=12, ha='center', va='center')
    
    def add(self, x, y, name):
        self.queue.put((x, y, name))
    
    def show(self):
        plt.show()