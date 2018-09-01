#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### Potential Feature List: determining what features should be used
#Below is the feature list I considered originally veruses the final product
original_list = ['poi',
                 'salary', 
                 'total_stock_value',
                 'from_poi_to_this_person',
                 'from_this_person_to_poi', 
                 'shared_receipt_with_poi', 
                 'bonus' ] 

features_list = [
        'poi',
#        'salary', 
#        'total_stock_value',
#        'frac_from_poi_to_this_person',
#        'frac_from_this_person_to_poi', 
        'frac_shared_receipt_with_poi', 
        'bonus' 
        ] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
#=================================================
# Step 1. Preliminary Feature and Dataset Analysis
#=================================================

def NaN_counter(data, feature):
    """Arguments:
        data, unpickled dict of dicts that stores the Enron information
        feature, str of the feature name
    Returns:
        new_data, list of instances without NaN
        NaN_Count, number of missing data entries"""
    new_data = []
    NaN_Count = 0
    for name in data.keys():
        if data[name][feature] != 'NaN':
            new_data.append(data[name][feature])
        else:
            NaN_Count += 1
    return new_data, NaN_Count
        
def feature_analysis(feature,data):
    """Arguments:
        feature, str of the feature name 
        data, unpickled dict of dicts that stores the Enron Information
        variable, str of whether it is numerical or categorical
    Returns:
        feature_info, dict of important values mapped to string
        e.g. {"Max" = 2450, "Min" = 2100}
        """
    from collections import Counter
    feature_info = {}
    values, NaN_count = NaN_counter(data,feature)
    feature_info["NaN count"] = NaN_count
    if feature == "email_address":
        feature_info["No. of addresses"] = len(values)
    else:
        values_freq = Counter(values)
        feature_info["Max"] = max(values_freq.keys())
        feature_info["Min"] = min(values_freq.keys())
    return feature_info

def print_feature(feature, feature_info):
    """Arguments:
        feature, str of the feature name
        feature_info, dict of important values for the feature
    Returns
        None, but prints out the information"""
    print("\n",str.upper(feature))
    for info in feature_info.keys():
        print("{}: {}".format(info, feature_info[info]))
    return None

def data_analysis(feature_list, data):
    """Arguments:
        feature_list, list of feature names as strings
        data, unpickled dict of dicts that stores the Enron Information
    Returns:
        None, but prints out the following information:
            1. Number of entries in the dataset
            2. Feature name and relevant info for each feature in data
            3. Features with a proportionally high number of missing values"""
    
    total_people = len(data.keys())
    print("Total number of entries:",total_people)
    #Finding the total number of people
    missing_info = []
    for feature in feature_list:
        feature_info = feature_analysis(feature, data)
        if feature_info["NaN count"] > total_people / 2:
            missing_info.append(feature)
        print_feature(feature, feature_info)
        #Prints the feature info and keeps track of features with over half NaN values
    print("\nFeatures with over half of the observations missing:\n",missing_info)
    #Can input feature to be analyzed (Example below)
    return None

##Example Runthrough of the above code:
#    
#data_analysis(original_list, data_dict)
#    
#=================================================
# Task 2. Outlier Removal
#=================================================
def drawScatterPlot(features, poi, mark_poi=False, name=("image.png",False), f1_name="feature 1", f2_name="feature 2"):
    """Arguments:
        features, list of arrays containing values for two features, n by 2
        poi, list of values for poi feature
        mark_poi, bol that defaults as False, and marks pois if True
        name, tuple of two elements:
            name[0] = str of the name the file will be saved as
            name[1] = bol that defaults as False, and will save the plot if True
        f1_name, str of the name for the first feature
        f2_name, str of the name for the second feature
    Returns:
        nothing, but graphs a scatter plot of the two features
        """
    import matplotlib.pyplot as plt
    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    for ii, pp in enumerate(features):
        plt.scatter(features[ii][0], features[ii][1], color = "b")

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(features):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    if name[1]:
        plt.savefig(name[0])
    plt.show()
    return None

##NOTE##
# Above function has features, poi meaning that I needed to input my dataset 
# and get labels, features from featureformat already. The only modification I
# made to this function from previous was having name be a tuple, so I could 
# save plots to the directory for later use.

# Example assuming features and labels are already split:
#drawScatterPlot(features,
#                labels,
#                mark_poi = False,
#                name = ("salary_vs_stock.pdf", True),
#                f1_name = "salary", 
#                f2_name = "stock")
# And it saves a pdf of the scatterplot in the same directory
#
#From plotting various features, identified TOTAL and since it is not a valid
#observation removed it below
data_dict.pop("TOTAL")


#============================================
# Step 3. Creating New Features
#============================================

def divide(value1, value2):
    """Arguments:
        value1, str of int
        value2, str of int
    Returns:
        new_val, value/value2 if executable"""
    if value1 == 'NaN' or value2 == 'NaN':
        return 'NaN'
    else:
        new_val = float(value1)/int(value2)
        return new_val
    
def add_feature(data,name,method):
    """Arguments:
        data, unpickled dict of dicts containing the information
        name, str of the feature name
        method, tuple of 3 elements:
            method[0] = feature_1
            method[1] = feature_2
            method[2] = function to be applied
            Ex: ("salary", "total_stock_value", divide)
    Returns:
        nothing, but name is now a feature for each key in data is added"""
    for key in data.keys():
        value_1 = data[key][method[0]] 
        value_2 = data[key][method[1]]
        new_value = method[2](value_1,value_2)
        data[key][name] = new_value
    return None

#Added the fractions of messages involving POIs (as stated in the report) below

method = [("from_this_person_to_poi", "from_messages", divide),
           ("from_poi_to_this_person", "to_messages", divide),
           ("shared_receipt_with_poi", "to_messages", divide)]

new_features = ["frac_from_this_person_to_poi",\
                "frac_from_poi_to_this_person",\
                "frac_shared_receipt_with_poi"]

for i in range(len(new_features)):
    add_feature(data_dict, new_features[i], method[i])

##Looking at new features:
#data_analysis(new_features, data_dict)
##Who can have more shared_receipts than messages sent to them?
#for key in data_dict.keys():
#    val = data_dict[key]['frac_shared_receipt_with_poi']
#    if val != 'NaN':
#        if data_dict[key]['frac_shared_receipt_with_poi'] > 1:
#            print(key)
##Name = GLISAN JR BEN F

#####Storing Data and extracting features/labels
my_dataset = data_dict
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
#####Storing Data and extracting features/labels

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#============================================
# Step 4. PCA and Feature Selection
#============================================ 
## Importing Operations/Pipeline
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


# Classifier List (Excluding SVM because disappointing reuslts)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#Examples of setting up classifiers (These were optimal)
    
bayes_pipe = Pipeline([ 
        ("pca", PCA() ),
        ("classifier", GaussianNB() )
                     ])
    
tree_pipe = Pipeline([ 
        ("pca", PCA() ), 
        ("classifier", DecisionTreeClassifier(random_state = 2) )
                     ])

knn_pipe = Pipeline([ 
        ("pca", PCA() ),
        ("scale", MinMaxScaler() ), 
        ("classifier", KNeighborsClassifier() )
                     ])

randomforest_pipe = Pipeline([ 
        ("pca", PCA() ),
        ("classifier", RandomForestClassifier(random_state = 2))
                      ])

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

#============================================
# Step 5. Algorithm Decision-Making
#============================================
import warnings
warnings.filterwarnings("ignore")

def clf_evaluation(clf, features_train, features_test, labels_train, labels_test):
    """Arguments:
        clf, the classifier 
        features_train/features_test, lists of the feature values
        labels_train/labels_test, lists of the labels (POI/nonPOI)
    Returns:
        tuple of 5 elements (all numeric):
            tuple[0], training time in seconds
            tuple[1], testing time in seconds
            tuple[2], accuracy of classifier
            tuple[3], recall of classifier
            tuple[4], precision of classifier"""
    from sklearn.metrics import recall_score, precision_score, accuracy_score
    import time as t
    #fitting and getting results
    t0 = t.time()
    clf.fit(features_train, labels_train)
    training_time = round(t.time() - t0, 5)
    t1 = t.time()
    labels_pred = clf.predict(features_test)
    testing_time = round(t.time() - t1, 5)
    accuracy = round(accuracy_score(labels_test, labels_pred), 4)
    recall = round(recall_score(labels_test,labels_pred), 4)
    precision = round(precision_score(labels_test,labels_pred), 4)
    return (training_time, testing_time, accuracy, recall, precision)

def classifier_cross_validation(clf, clf_name,features, labels):
    """Arguments:
        clf, the classifier
        clf_name, string of classifier name
        features, list of features to be used
        labels, list of labels to be used
    Returns:
        nothing, but evaluates and prints the results of:
            1. A 30% Train-Test Split 
            2. 10-Fold cross validations
        For each, returns training/testing time, accuracy, precision, and recall"""
    from sklearn.cross_validation import train_test_split, KFold
    import numpy as np
    print("\n{} Classifier\n".format(clf_name))
    result_names = ["Training time (s)", "Testing time (s)", "Accuracy", \
                    "Recall", "Precision"]
    #train_test_split
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)
    results = clf_evaluation(clf, features_train, features_test, \
                             labels_train, labels_test)
    print("     Result of 30% Test set size:")
    print_results(result_names, results)
    
    #kfold
    total_results = np.zeros(5)
    kf = KFold(len(features), 10)
    for train_i, test_j in kf:
        features_train, features_test, labels_train, labels_test = [[],[],[],[]]
        for i in train_i:
            features_train.append(features[i])
            labels_train.append(labels[i])
        for j in test_j:
            features_test.append(features[j])
            labels_test.append(labels[j])
        results = clf_evaluation(clf, features_train, features_test, \
                                 labels_train, labels_test)
        total_results = total_results + np.array(results)
    avg_results = np.round(total_results * 0.1, 4)
    print("     Result of 10-Fold cross validation (avg):")
    print_results(result_names, avg_results)
    return None

def print_results(labels, values):
    """Arguments:
        labels, list of strings
        values, list of numerics
    Returns:
        nothing, but is a helper print function"""
    for i in range(len(labels)):
        print("{}: {}".format(labels[i], values[i]))
    return None
    
#Example of proper way to run above code
#classifier_cross_validation(tree_pipe, "PCA Decision Tree", features, labels)

#============================================
# Step 5b. Fine-Tuning Parameters
#============================================
def tester_scoring(clf, features, labels):
    """Slightly modified version of tester function in tester.py,
    returns the f1_score after 1000 fold stratified shuffle split.
    
    Why? To use as the scoring function for GridSearch so it optimizes
    the tester.py results when comparing the classifier at different inputs,
    since f1_score is roughly equally weighting of precision and recall"""
    from sklearn.cross_validation import StratifiedShuffleSplit
    cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print("Warning: Found a predicted label not == 0 or 1.")
                print("All predictions should take value 0 or 1.")
                print("Evaluating performance for processed predictions:")
                break

    f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
    return f1

from sklearn.model_selection import GridSearchCV

###         Bayes: No real tuning available
#features used: total_stock_value,bonus
bayes_clf = bayes_pipe

###         Decision Tree
#features used: frac_shared_receipt_with_poi,bonus
tree_params = {'classifier__criterion': ('gini', 'entropy'),
               'classifier__min_samples_split': (2,3,4),
               'classifier__max_depth' : (None,2,3,4)
               }
grid_tree_clf = GridSearchCV(tree_pipe,
                             tree_params,
                             scoring = tester_scoring)
###Warning, will take a while

#grid_tree_clf.fit(features, labels)
#print(grid_tree_clf.best_params_)
#Output: criterion = entropy, depth = None, min_samples_split = 2
tree_clf = Pipeline([ 
        ("pca", PCA() ), 
        ("classifier", DecisionTreeClassifier(criterion = 'entropy',
                                              min_samples_split = 2,
                                              max_depth = None,
                                              random_state = 2) )
                     ])

###         K-Nearest Neighbors
#features used: bonus
knn_params = {'classifier__n_neighbors': (5,6,7),
              'classifier__weights': ('uniform', 'distance'),
              'classifier__p': (1,2,3)}
grid_knn_clf = GridSearchCV(knn_pipe,
                            knn_params,
                            scoring = tester_scoring)
###Warning, will take a while
#grid_knn_clf.fit(features, labels)
#print(grid_knn_clf.best_params_)
#Output: n_neighbors = 5, p = 1, weights = distance
knn_clf = Pipeline([ 
        ("pca", PCA() ),
        ("scale", MinMaxScaler() ), 
        ("classifier", KNeighborsClassifier(p = 1,
                                            weights = 'distance') )
                     ])

###         Random Forest
#features used: frac_shared_receipt_with_poi, bonus
randomforest_params = {'classifier__n_estimators': (5,10),
                       'classifier__min_samples_split': (2,3,4),
                       'classifier__max_depth': (None,2)
                       }
grid_randomforest_clf = GridSearchCV(randomforest_pipe,
                                     randomforest_params,
                                     scoring = tester_scoring)
###Wow, this took 716 seconds-over 10 minutes. Not scalable at all to larger datasets
#grid_randomforest_clf.fit(features, labels)
#print(grid_randomforest_clf.best_params_)
#Options: max_depth = None, min_samples_split = 2, n_estimators = 5
randomforest_clf = Pipeline([ 
        ("pca", PCA() ),
        ("classifier", RandomForestClassifier(n_estimators = 5,
                                              min_samples_split = 2,
                                              max_depth = None,
                                              random_state = 2))
                      ])
        
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
#=========================
# Depositing the classifier
#=========================

clf = tree_clf


dump_classifier_and_data(clf, my_dataset, features_list)