import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

activity = ["LAYING", "SITTING", "STANDING", "WALKING", "WALKING_DOWNSTAIRS", "WALKING_UPSTAIRS"]
dir = os.path.join("Combined", "Train")
time = 10
freq = 50  

linear_samples = []
for folder in activity:
    path = os.path.join(dir, folder)
    files = os.listdir(path)
    file_selected = os.path.join(path, files[0])
    
    df = pd.read_csv(file_selected)    
    df50 = df[:time * freq]
     
    
    acc_X = df50['accx'].values
    acc_Y = df50['accy'].values
    acc_Z = df50['accz'].values
       
    
    acc = np.vstack(((acc_X)**2,(acc_Y)**2,(acc_Z)**2)).T    
    linear_acc = np.sum(acc, axis=1)
    linear_samples.append(linear_acc)
       
linear_samples = np.array(linear_samples)

linear_samples = linear_samples.T
plt.figure(figsize=(18, 8))
for i in range(len(activity)):
    plt.subplot(2, 3, i + 1)
    plt.plot(linear_samples[:, i])
    plt.title(activity[i])
    plt.xlabel('Samples')
    plt.ylabel('Linear Acceleration')

plt.tight_layout()
plt.show()

static= ['LAYING', 'SITTING', 'STANDING']
dynamic = ['WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']

static_avg_acc = [np.mean(linear_samples[i]) for i in range(3)]
dynamic_avg_acc = [np.mean(linear_samples[i+3]) for i in range(3)]

print("Avg for Static Activities:", static_avg_acc)
print("Avg for Dynamic Activities:", dynamic_avg_acc)

print(("-----"*20))

if max(static_avg_acc) < min(dynamic_avg_acc):
    print("No machine learning is required. Only from threshold we can check.")
else:
    print("A machine learning model may be needed.")
    
print(("-----"*20))




