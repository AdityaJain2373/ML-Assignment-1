import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

paths = {
    'Laying' : r'ML-Assignment-1\UCI HAR Dataset\Combined\Train\LAYING\Subject_5.csv',
    'Sitting' : r'ML-Assignment-1\UCI HAR Dataset\Combined\Train\SITTING\Subject_5.csv',
    'Standing' : r'ML-Assignment-1\UCI HAR Dataset\Combined\Train\STANDING\Subject_5.csv',
    'Walking' : r'ML-Assignment-1\UCI HAR Dataset\Combined\Train\WALKING\Subject_5.csv',
    'Walking_Downstairs' : r'ML-Assignment-1\UCI HAR Dataset\Combined\Train\WALKING_DOWNSTAIRS\Subject_5.csv',
    'walking_Upstairs' : r'ML-Assignment-1\UCI HAR Dataset\Combined\Train\WALKING_UPSTAIRS\Subject_5.csv'
}

l = {}

for activity in paths:
    data = pd.read_csv(paths[activity])
    data["Linear Acceleration"] = np.sqrt((data["accx"][:500])**2 + (data['accy'][:500])**2 + (data['accz'][:500])**2)
    l[activity] = np.mean(data["Linear Acceleration"])
    print(f" {activity} : {np.mean(data["Linear Acceleration"])}")

print()
standard_deviation_acc = np.std(list(l.values()))

print(standard_deviation_acc)

