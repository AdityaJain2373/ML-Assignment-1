
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'


activity = ["LAYING" , "SITTING","STANDING","WALKING", "WALKING_DOWNSTAIRS","WALKING_UPSTAIRS"]
dir = os.path.join("Combined","Train")
time = 10
offset = 100
freq = 50

samples = []
for folders in activity:
    path = os.path.join(dir, folders)
    file = os.listdir(path)
    
    file_selected = os.path.join(path ,file[0])
    
    df = pd.read_csv(file_selected)
    
    df50 = df[ : time * freq]
    
    samples.append(df50.values)
    
samples = np.array(samples)


plt.figure(figsize=(18,6))
for i, sample in enumerate(samples):
    plt.subplot(2,3,i+1)
    plt.plot(sample)
    plt.title(activity[i],fontsize=8)
plt.tight_layout()
plt.show()


        
    
    
