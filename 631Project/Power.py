import csv
import numpy as np
import scipy.interpolate


class Power:
    # Handles the measurements of antenna power
    # To collect power, in Linux Command Line, type: rtl_power -f 914:5M:915.5M:10k -g -10 -i 1s -e 30s antenna.csv
    def __init__(self, frame_count, run_name):
        self.peak_power = []
        with open('{}.csv'.format(run_name), newline='') as csv_file:
            reader = csv.reader(csv_file, delimiter=',', quotechar='|')
            for row in reader:
                # Assuming the peak power is the power at the intended frequency
                self.peak_power.append(max([float(i) for i in row[6:]]))

        self.peak_power = np.asarray(self.peak_power)

        # Convert peak_power to numpy array
        self.peak_power -= np.amax(self.peak_power)  # Assumes min will actually be a node with effectively no power.

        # Interpolate peak_power to match number of video frames
        x = np.arange(0, frame_count + 50, (frame_count + 50) / len(self.peak_power))
        interpolation_function = scipy.interpolate.interp1d(x, self.peak_power)
        x_new = np.arange(frame_count)
        self.peak_power = interpolation_function(x_new)