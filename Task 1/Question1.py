import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

paths = {
    'Laying' : r'ML-Assignment-1\UCI HAR Dataset\Combined\Train\LAYING\Subject_3.csv',
    'Sitting' : r'ML-Assignment-1\UCI HAR Dataset\Combined\Train\SITTING\Subject_3.csv',
    'Standing' : r'ML-Assignment-1\UCI HAR Dataset\Combined\Train\STANDING\Subject_3.csv',
    'Walking' : r'ML-Assignment-1\UCI HAR Dataset\Combined\Train\WALKING\Subject_3.csv',
    'Walking_Downstairs' : r'ML-Assignment-1\UCI HAR Dataset\Combined\Train\WALKING_DOWNSTAIRS\Subject_3.csv',
    'walking_Upstairs' : r'ML-Assignment-1\UCI HAR Dataset\Combined\Train\WALKING_UPSTAIRS\Subject_3.csv'
}

fig, axes = plt.subplots(nrows= 2, ncols= 3, figsize = (15,10))
axes = axes.flatten()

for ax, (activity_name, file_path) in zip(axes,paths.items()):
    activity_data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    ax.plot(activity_data[:500,0], label='accx')
    ax.plot(activity_data[:500,1], label='accy')
    ax.plot(activity_data[:500,2], label='accz')

    ax.set_title(activity_name)
    ax.set_xlabel('Time')
    ax.set_ylabel('Acceleration')
    ax.legend()

plt.tight_layout()
plt.show()


