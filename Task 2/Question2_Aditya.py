from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from Question1_Aditya import data_analysis,tsfel_data

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


def evaluate_tree_depths(X_train, y_train, X_test, y_test, min_depth=2, max_depth=8):
    depths = range(min_depth, max_depth + 1)
    accuracies = []

    for depth in depths:
        model = DecisionTreeClassifier(max_depth=depth)
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    plt.figure(figsize=(10, 6))
    plt.plot(depths, accuracies, marker='o')
    plt.title('Decision Tree Accuracy vs. Depth')
    plt.xlabel('Depth of Tree')
    plt.ylabel('Accuracy')
    plt.xticks(depths)
    plt.grid(True)
    plt.show()

    print(depths, accuracies)

X_train,y_train,X_test,y_test = data_analysis(train_paths,test_paths)
tsfel_train,tsfel_test = tsfel_data(X_train,X_test,y_train,y_test)
evaluate_tree_depths(X_train,y_train,X_test,y_test)
evaluate_tree_depths(tsfel_train,y_train,tsfel_test,y_test)
