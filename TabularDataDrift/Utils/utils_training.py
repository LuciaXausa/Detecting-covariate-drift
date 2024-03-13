# for training the random forest classifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics



def trainingRandomForest(seed_dataset, df, labelCol) :
    '''Function to split dataset and train random Forest for classification'''
    X = df.iloc[:, :-1]      # features DataFrame (the last column is label so we don't take that)
    y = df[labelCol]       # target



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state = seed_dataset)                        # train size is 0.4
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5,random_state = seed_dataset)                  # test and validation size is 0.3
    
    forest = RandomForestClassifier(n_estimators=50, random_state=seed_dataset)

    # Fitting a model and making predictions
    forest.fit(X_train,y_train)
    predictions = forest.predict(X_test)
    averageAcc = metrics.accuracy_score(y_test, predictions)
    return X_train, X_val, X_test, y_val, y_test, averageAcc, forest


def compute_accuracy(X_val, y_val, forest):
    '''accuracy of val'''
    predictions_val = forest.predict(X_val)
    accuracy = metrics.accuracy_score(y_val, predictions_val)

    return accuracy 
