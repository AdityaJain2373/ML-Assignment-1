import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
import seaborn as sns
import tsfel

train_paths = {
    'Laying' : r'ML-Assignment-1\UCI HAR Dataset\Combined\Train\LAYING\Subject_3.csv',
    'Sitting' : r'ML-Assignment-1\UCI HAR Dataset\Combined\Train\SITTING\Subject_3.csv',
    'Standing' : r'ML-Assignment-1\UCI HAR Dataset\Combined\Train\STANDING\Subject_3.csv',
    'Walking' : r'ML-Assignment-1\UCI HAR Dataset\Combined\Train\WALKING\Subject_3.csv',
    'Walking_Downstairs' : r'ML-Assignment-1\UCI HAR Dataset\Combined\Train\WALKING_DOWNSTAIRS\Subject_3.csv',
    'walking_Upstairs' : r'ML-Assignment-1\UCI HAR Dataset\Combined\Train\WALKING_UPSTAIRS\Subject_3.csv'
}

test_paths = {
    'Laying' : r'ML-Assignment-1\UCI HAR Dataset\Combined\Test\LAYING\Subject_4.csv',
    'Sitting' : r'ML-Assignment-1\UCI HAR Dataset\Combined\Test\SITTING\Subject_4.csv',
    'Standing' : r'ML-Assignment-1\UCI HAR Dataset\Combined\Test\STANDING\Subject_4.csv',
    'Walking' : r'ML-Assignment-1\UCI HAR Dataset\Combined\Test\WALKING\Subject_4.csv',
    'Walking_Downstairs' : r'ML-Assignment-1\UCI HAR Dataset\Combined\Test\WALKING_DOWNSTAIRS\Subject_4.csv',
    'walking_Upstairs' : r'ML-Assignment-1\UCI HAR Dataset\Combined\Test\WALKING_UPSTAIRS\Subject_4.csv'
}

def make_data(path):
    final_data = []
    labels = []
    mapping = {label: idx for idx, label in enumerate(path.keys())}

    for activity,path in path.items():
        data = pd.read_csv(path, nrows=500)
        data["label"] = mapping[activity]
        labels.append(data['label'])
        data['acc'] = np.sqrt(data['accx']**2 + data['accy']**2 + data['accz']**2)
        final_data.append(data[["accx",'accy','accz','acc','label']])

    return final_data,labels



def data_analysis(train_paths,test_paths):

    activity_mapping = {
    0: 'LAYING',
    1: 'WALKING',
    2: 'WALKING_UPSTAIRS',
    3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING',
    5: 'STANDING'
    }

    train_data, train_labels = make_data(train_paths)

    X_train = pd.concat(train_data, ignore_index=True)
    y_train = np.concatenate(train_labels)

    X_train, y_train = shuffle(X_train, y_train, random_state=420)

    train_combined = X_train.copy()
    train_combined["activity"] = y_train
    train_combined['activity'] = train_combined['activity'].map(activity_mapping)

    X_train_final = train_combined[["accx",'accy','accz','acc']]  
    y_train_final = train_combined["activity"]

    test_data, test_labels = make_data(test_paths)

    X_test = pd.concat(test_data, ignore_index=True)
    y_test = np.concatenate(test_labels)

    test_combined = X_test.copy()
    test_combined["activity"] = y_test
    test_combined['activity'] = test_combined['activity'].map(activity_mapping)

    X_test_final = test_combined[["accx",'accy','accz','acc']]  
    y_test_final = test_combined["activity"]

    return (X_train_final,y_train_final,X_test_final,y_test_final)


def tsfel_data(X_train_final,X_test_final,y_train_final,y_test_final):

    cfg_file = tsfel.get_features_by_domain("statistical")

    selected_features = {
    'statistical': {
        'root_mean_square': cfg_file['statistical']['Root mean square'],
        'Std': cfg_file['statistical']['Standard deviation'],
        'Mean': cfg_file['statistical']['Mean'],
        'ECDF': cfg_file['statistical']['ECDF']
        }
    }

    X_tsfel_train = tsfel.time_series_features_extractor(selected_features, X_train_final,window_size=1, verbose=0)
    X_tsfel_test = tsfel.time_series_features_extractor(selected_features, X_test_final, window_size=1,verbose=0)

    pca = PCA(n_components = 2)
    X_tsfel_train_pca = pca.fit_transform(X_tsfel_train)
    X_tsfel_test_pca = pca.transform(X_tsfel_test)

    pca_train_df = pd.DataFrame(data=X_tsfel_train_pca, columns=["PCA_1", "PCA_2"])
    pca_test_df = pd.DataFrame(data=X_tsfel_test_pca, columns=["PCA_1", "PCA_2"])

    pca_train_df["Label"] = y_train_final.reset_index(drop=True)
    pca_test_df["Label"] = y_test_final.reset_index(drop=True)

    pca_train_df, pca_test_df = shuffle(pca_train_df, pca_test_df, random_state=420)
    pca_train_df = pca_train_df.drop("Label",axis ="columns")
    pca_test_df = pca_test_df.drop("Label",axis = "columns")

    return(pca_train_df, pca_test_df)



def Model(X_train_final, y_train_final, X_test_final, y_test_final):

    model = DecisionTreeClassifier()
    model.fit(X_train_final, y_train_final)

    y_pred = model.predict(X_test_final)

    print("Test Accuracy:", accuracy_score(y_test_final, y_pred))
    print("Test Classification Report:")
    print(classification_report(y_test_final, y_pred))

    conf_matrix = confusion_matrix(y_test_final, y_pred)
    sns.heatmap(conf_matrix,annot = True)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

if __name__ == "__main__":
    X_train_final, y_train_final, X_test_final, y_test_final = data_analysis(train_paths, test_paths)
    pca_train_df, pca_test_df = tsfel_data(X_train_final, X_test_final,y_train_final,y_test_final)
    Model(pca_train_df,y_train_final,pca_test_df,y_test_final)
    Model(X_train_final,y_train_final,X_test_final,y_test_final)


