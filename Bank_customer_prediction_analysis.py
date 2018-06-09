
# coding: utf-8

# In[78]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# # DEFINING FUNCTIONS

# FUNCTION: Spliting data into Training and Test set and normalizing each feature 
# 
# INPUT: entire data set, cleaned dataset
# 
# OUTPUT:Training and Testing sets

# In[36]:


def train_test(work,new_data):
    label=(work.y=='no').astype(int)
    label=pd.DataFrame(label, columns=['y']);
    label.rename(columns={"y" : "label"}, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split( new_data, label['label'], test_size=0.25, random_state=100)
    #Standard scaler normalizes data column wise
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test 


# FUNCTION: Performs PCA for dimensionality reduction on the dataset
# 
# INPUT: cleaned dataset, number of principle components to reduce the dimensionality to
# 
# OUTPUT:Dataset with reduced dimensionalty 

# In[37]:


def perform_pca(new_data, components):
    pca = PCA(n_components=components)
    pca_new_data = pca.fit_transform(new_data)
    pca_new_data = pd.DataFrame(data = pca_new_data, columns = ['principal component 1', 'principal component 2']);
    return pca_new_data


# FUNCTION: Performs SMOTE for data augmentation
# 
# INPUT: training and testing datasets
# 
# OUTPUT: augmented training and testing sets

# In[38]:


def applying_SMOTE(X_train, y_train,X_test, y_test):
    sm = SMOTE(kind='regular')
    X_res_train, y_res_train = method.fit_sample(X_train, y_train)
    X_res_test, y_res_test = method.fit_sample(X_test, y_test)
    return X_res_train, y_res_train,X_res_test, y_res_test


# FUNCTION: Prints metrics 
# 
# INPUT: testing set and predicted labels
# 
# OUTPUT: prints the cofusion matrix and F1 score matrix

# In[56]:


def print_metrics(test, pred_label, title):
    print("The metrics for " + title + "\n");
    cm = confusion_matrix(test, pred_label)
    print("Confusion Matrix:\n");
    print(pd.DataFrame(cm))
    f1 = f1_score(test, pred_label, average='weighted')
    print("\nF1 score is: ", round((f1 * 100), 2), "%");
    print(classification_report(test,pred_label))
    return f1


# In[57]:


def print_metrics_classifier(title, score_train, score_test, cm, f1, cr):   
    print_bold(title);
    print("\nConfusion Matrix");
    print(pd.DataFrame(cm))
    print("\nTraining Accuracy: ", round(score_train * 100, 2), "%");
    print("\nTesting Accuracy: ", round(score_test * 100, 2), "%");
    print("\nF1 score is: ", round(f1, 5));
    print("\nClassification Report");
    print(cr);


# In[58]:


def print_bold(data):
    print('\033[1m' + str(data) + '\033[0m')
    return


# Class: classifier_info
# 
# Description : The classifier_info class has all methods to classify and print metrics

# In[59]:


class classifier_info:
    
    def __init__(self, clf, X_train, X_test, y_train, y_test, title):
        self.clf = clf
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.title = title
        
    def classify(self):
        self.clf.fit(self.X_train, self.y_train);
        self.train_score = self.clf.score(self.X_train, self.y_train)
        self.test_score = self.clf.score(self.X_test, self.y_test)
        self.y_pred = self.clf.predict(self.X_test)
        self.cr = classification_report(self.y_test, self.y_pred)
        self.cm = metrics.confusion_matrix(self.y_test, self.y_pred)
        self.f1 = metrics.f1_score(self.y_test, self.y_pred, average='weighted')
        
        
        
    def metrics(self, PRINT_FLAG="yes"):
        x = PrettyTable()
        x.field_names = ["Classifier", "Training Score", "Testing Score", "F1-score"]
        print_metrics_classifier(self.title, self.train_score, self.test_score, self.cm, self.f1, self.cr)
        if(PRINT_FLAG == "yes"):
            x.add_row([self.title, str(round((self.train_score * 100), 2)) + "%", str(round((self.test_score * 100), 2)) + "%" , str(round(self.f1, 5))])
    
  


# # READING DATA

# In[11]:


data= pd.read_csv("bank-additional.csv")
#DISPLAYING DATA
data.head()
work = data


# # PLOTTING HISTOGRAMS FOR EACH FEATURE/CATEGORY

# In[90]:


fig=plt.figure(figsize=(15,5))
ax = fig.gca()
data.age.value_counts().plot(kind='bar',width=0.9);
plt.suptitle('Histogram for types of Age', fontsize=20)


fig=plt.figure(figsize=(15,5))
ax = fig.gca()
data.job.value_counts().plot(kind='bar',width=0.9);
plt.suptitle('Histogram for types of Job', fontsize=20)


fig=plt.figure(figsize=(8,5))
ax = fig.gca()
data.marital.value_counts().plot(kind='bar',width=0.9);
plt.suptitle('Histogram for Marital status', fontsize=20)

fig=plt.figure(figsize=(8,5))
ax = fig.gca()
data.education.value_counts().plot(kind='bar',width=0.9);
plt.suptitle('Histogram for Education status', fontsize=20)



fig=plt.figure(figsize=(8,5))
ax = fig.gca()
data.default.value_counts().plot(kind='bar',width=0.9);
plt.suptitle('Histogram for credit defaulters present in data', fontsize=20)


fig=plt.figure(figsize=(8,5))
ax = fig.gca()
data.housing.value_counts().plot(kind='bar',width=0.9);
plt.suptitle('Histogram for housing present in data', fontsize=20)


fig=plt.figure(figsize=(8,5))
ax = fig.gca()
data.loan.value_counts().plot(kind='bar',width=0.9);
plt.suptitle('Histogram for loan present in data', fontsize=20)

fig=plt.figure(figsize=(8,5))
ax = fig.gca()
data.contact.value_counts().plot(kind='bar',width=0.9);
plt.suptitle('Histogram for contact present in data', fontsize=20)


fig=plt.figure(figsize=(10,5))
ax = fig.gca()
data.month.value_counts().plot(kind='bar',width=0.9);
plt.suptitle('Histogram for month present in data', fontsize=20)



fig=plt.figure(figsize=(8,5))
ax = fig.gca()
data.day_of_week.value_counts().plot(kind='bar',width=0.9);
plt.suptitle('Histogram for day of week present in data', fontsize=20)


fig=plt.figure(figsize=(8,5))
ax = fig.gca()
data.campaign.value_counts().plot(kind='bar',width=0.9)
plt.suptitle('Histogram for campaign present in data', fontsize=20)


fig=plt.figure(figsize=(8,5))
ax = fig.gca()
data.pdays.value_counts().plot(kind='bar',width=0.9)
plt.suptitle('Histogram for pdays present in data', fontsize=20)


fig=plt.figure(figsize=(10,5))
ax = fig.gca()
data.previous.value_counts().plot(kind='bar',width=0.9)
plt.suptitle('Histogram for previous contacted donators present in data', fontsize=20)

fig=plt.figure(figsize=(8,5))
ax = fig.gca()
data.poutcome.value_counts().plot(kind='bar',width=0.9);
plt.suptitle('Histogram for poutcome present in data', fontsize=20)


# # DATA CLEANING

# In[13]:


#Removing 'job' unknown values 
work = work[work.job != 'unknown']
#Removing 'marital' unknown values
work = work[work.marital != 'unknown']
#Removing 'education' illiterate values
work = work[work.education != 'illiterate']
#Removing 'default' yes values
work = work[work.default != 'yes']
#Removing 'housing' unknown values
work = work[work.housing != 'unknown']
#Removing 'loan' unknown values
work = work[work.loan != 'unknown']


# Getting one-hot data for 'age'

# In[14]:




a1 =  (work.age < 30).astype(int)
new_data = pd.DataFrame(a1, columns=['age'])
new_data.rename(columns={"age":"age_adult"}, inplace=True)

a1 =  ((work.age >= 30) & (work.age<60)).astype(int)
new_data = pd.concat([new_data, a1], axis=1)
new_data.rename(columns={"age":"age_middle"}, inplace=True)

a1 =  (work.age >= 60).astype(int)
new_data = pd.concat([new_data, a1], axis=1)
new_data.rename(columns={"age":"age_old"}, inplace=True)


# Getting one-hot data for 'job'

# In[15]:


new_data = pd.concat([new_data, pd.get_dummies(work.job, prefix="job")], axis=1);


# Getting one-hot data for 'marital' 

# In[16]:


new_data = pd.concat([new_data, pd.get_dummies(work.marital, prefix="marital")], axis=1)


# Getting one-hot data for 'education' 

# In[17]:


new_data = pd.concat([new_data, pd.get_dummies(work.education, prefix="education")], axis=1)


# Getting one-hot data for 'default' 

# In[18]:


new_data = pd.concat([new_data, pd.get_dummies(work.default, prefix="default")], axis=1)


# Getting one-hot data for 'housing'

# In[19]:


new_data = pd.concat([new_data, pd.get_dummies(work.housing, prefix="housing")], axis=1)


# Getting one-hot data for 'loan'

# In[20]:


new_data = pd.concat([new_data, pd.get_dummies(work.loan, prefix="loan")], axis=1)


# Getting one-hot data for 'contact'

# In[21]:


new_data = pd.concat([new_data, pd.get_dummies(work.contact, prefix="contact")], axis=1)


# Getting one-hot data for 'month'

# In[22]:


new_data = pd.concat([new_data, pd.get_dummies(work.month, prefix="month")], axis=1)


# Getting one-hot data for 'day_of_week'

# In[23]:


new_data = pd.concat([new_data, pd.get_dummies(work.day_of_week, prefix="day_of_week")], axis=1)


# Getting one-hot data for 'poutcome'	

# In[24]:


new_data = pd.concat([new_data, pd.get_dummies(work.poutcome, prefix="poutcome")], axis=1)


# In[25]:


new_data = pd.concat([new_data,work.campaign], axis=1)
new_data = pd.concat([new_data,work.previous], axis=1)
new_data = pd.concat([new_data,work['emp.var.rate']], axis=1)
new_data = pd.concat([new_data,work['cons.price.idx']], axis=1)
new_data = pd.concat([new_data,work['cons.conf.idx']], axis=1)
new_data = pd.concat([new_data,work.euribor3m], axis=1)
new_data = pd.concat([new_data,work['nr.employed']], axis=1)


# In[26]:


# the dimensions of the final matrix (dataPoints x features)
new_data.shape


# 
# 
# # KNN

# In[100]:


X_train, X_test, y_train, y_test = train_test(work,new_data)

neigh = KNeighborsClassifier(n_neighbors=15)
neigh.fit(X_train, y_train)

trainlabels=neigh.predict(X_train)
train_acc=accuracy_score(y_train,trainlabels)
# print("KNN Classification accuracy for Training: %f"% (train_acc))


newlabels = neigh.predict(X_test)
acc=accuracy_score(y_test,newlabels)
# print("KNN Classification accuracy for Testing: %f "% (acc))

output_f1= print_metrics(y_test, newlabels, 'KNN')

x = PrettyTable()
x.field_names = ["Classifier", "Training Score", "Testing Score", "F1-score"]
x.add_row(["KNN", str(round((train_acc * 100), 2)) + "%", str(round((acc * 100), 2)) + "%" , str(output_f1 *100) + "%"])
print(x)
print ("roc : ",roc_auc_score(y_test, newlabels))


y_true = y_test
y_pred = newlabels
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # Multilayer perceptron

# In[101]:


X_train, X_test, y_train, y_test = train_test(work,new_data)
   
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
mlp.fit(X_train,y_train)
newlabels = mlp.predict(X_test)

trainlabels=mlp.predict(X_train)
train_acc=accuracy_score(y_train,trainlabels)
# print("Multilayer perceptron Classification accuracy for Training: %f"% train_acc)

acc=accuracy_score(y_test,newlabels)
# print("Multilayer Perceptron Classification accuracy for Testing: %f"% acc)
output_f1 = print_metrics(y_test, newlabels, 'Multilayer Perceptron')

x = PrettyTable()
x.field_names = ["Classifier", "Training Score", "Testing Score", "F1-score"]
x.add_row(["Multilayer Perceptron", str(round((train_acc * 100), 2)) + "%", str(round((acc * 100), 2)) + "%" , str(output_f1 *100) + "%"])
print(x)
print ("roc : ",roc_auc_score(y_test, newlabels))


y_true = y_test
y_pred = newlabels
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# 
# 
# 
# # Perceptron

# In[102]:


X_train, X_test, y_train, y_test = train_test(work,new_data)
   
   
clf = Perceptron()
clf.fit(X_train,y_train )
newlabels=clf.predict(X_test)

trainlabels=clf.predict(X_train)
train_acc=accuracy_score(y_train,trainlabels)
# print("Perceptron Classification accuracy for Training: %f"% train_acc)
acc=accuracy_score(y_test,newlabels)
# print("Perceptron Classification accuracy for Testing: %f"% acc)
output_f1 = print_metrics(y_test, newlabels, 'Perceptron')

x = PrettyTable()
x.field_names = ["Classifier", "Training Score", "Testing Score", "F1-score"]
x.add_row(["Perceptron", str(round((train_acc * 100), 2)) + "%", str(round((acc * 100), 2)) + "%" , str(output_f1 *100) + "%"])
print(x)
print ("roc : ",roc_auc_score(y_test, newlabels))


y_true = y_test
y_pred = newlabels
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # PCA + linear SVM Classifier

# In[103]:




pca_new_data = perform_pca(new_data, 2)
X_train, X_test, y_train, y_test = train_test(work,pca_new_data)
    
    
clf = svm.SVC( C=1, kernel='linear')
clf.fit(X_train, y_train)
newlabels=clf.predict(X_test)


trainlabels=clf.predict(X_train)
train_acc=accuracy_score(y_train,trainlabels)
# print("PCA + SVM accuracy for Training: %f"% train_acc)

acc=accuracy_score(y_test,newlabels)
# print("PCA + SVM Classification accuracy for Testing: ",acc)

output_f1 = print_metrics(y_test, newlabels, 'SVM with PCA')
x = PrettyTable()
x.field_names = ["Classifier", "Training Score", "Testing Score", "F1-score"]
x.add_row(["SVM with PCA", str(round((train_acc * 100), 2)) + "%", str(round((acc * 100), 2)) + "%.f" , str(((output_f1) *100)) + "%"])
print(x)
print ("roc : ",roc_auc_score(y_test, newlabels))


y_true = y_test
y_pred = newlabels
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # linear SVM classifier

# In[104]:


X_train, X_test, y_train, y_test = train_test(work,new_data)

clf = svm.SVC( C=1, kernel='linear')
clf.fit(X_train, y_train)
newlabels=clf.predict(X_test)


trainlabels=clf.predict(X_train)
train_acc=accuracy_score(y_train,trainlabels)
# print("SVM accuracy for Training: %f"% train_acc)

acc=accuracy_score(y_test,newlabels)
# print("SVM Classification accuracy for Testing: ",acc)

output_f1 = print_metrics(y_test, newlabels, 'SVM with PCA')
x = PrettyTable()
x.field_names = ["Classifier", "Training Score", "Testing Score", "F1-score"]
x.add_row(["SVM with PCA", str(round((train_acc * 100), 2)) + "%", str(round((acc * 100), 2)) + "%" , str(output_f1 *100) + "%"])
print(x)
print ("roc : ",roc_auc_score(y_test, newlabels))


y_true = y_test
y_pred = newlabels
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # NAIVE BAYES 

# In[105]:


X_train, X_test, y_train, y_test = train_test(work,new_data)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
newlabels=gnb.predict(X_test)


trainlabels=gnb.predict(X_train)
train_acc=accuracy_score(y_train,trainlabels)
# print("naive bayes accuracy for Training: %f"% train_acc)

acc=accuracy_score(y_test,newlabels)
# print("naive bayes Classification accuracy for Testing: ",acc)

output_f1 = print_metrics(y_test, newlabels, 'Naive Bayes')

x = PrettyTable()
x.field_names = ["Classifier", "Training Score", "Testing Score", "F1-score"]
x.add_row(["Naive Bayes", str(round((train_acc * 100), 2)) + "%", str(round((acc * 100), 2)) + "%" , str(output_f1 *100) + "%"])
print(x)
print ("roc : ",roc_auc_score(y_test, newlabels))


y_true = y_test
y_pred = newlabels
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# 
# # PCA + Naive Bayes 

# In[106]:


pca_new_data = perform_pca(new_data, 2)
X_train, X_test, y_train, y_test = train_test(work,pca_new_data)

gnb = GaussianNB()

gnb.fit(X_train, y_train)
newlabels=gnb.predict(X_test)


trainlabels=gnb.predict(X_train)
train_acc=accuracy_score(y_train,trainlabels)
# print("PCA + Naive Bayes accuracy for Training: %f"% train_acc)

acc=accuracy_score(y_test,newlabels)
# print("PCA + Naive Bayes Classification accuracy for Testing: ",acc)

output_f1 = print_metrics(y_test, newlabels, 'naive bayes with PCA')

x = PrettyTable()
x.field_names = ["Classifier", "Training Score", "Testing Score", "F1-score"]
x.add_row(["naive bayes with PCA", str(round((train_acc * 100), 2)) + "%", str(round((acc * 100), 2)) + "%" , str(output_f1 *100) + "%"])
print(x)
print ("roc : ",roc_auc_score(y_test, newlabels))


y_true = y_test
y_pred = newlabels
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # Apply regular SMOTE + MLP

# In[88]:


pca_new_data = perform_pca(new_data, 2)
X_train, X_test, y_train, y_test = train_test(work,pca_new_data)
pca = PCA(n_components=2)

# X_res_train, y_res_train,X_res_test, y_res_test = apply_SMOTE(X_train, y_train,X_test, y_test)
method = SMOTE(kind='regular')
X_res_train, y_res_train = method.fit_sample(X_train, y_train)
# X_res_test, y_res_test = method.fit_sample(X_test, y_test)

mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
mlp.fit(X_res_train,y_res_train)
newlabels = mlp.predict(X_test)

trainlabels=mlp.predict(X_res_train)
train_acc=accuracy_score(y_res_train,trainlabels)
# print("Multilayer perceptron Classification accuracy for Training: %f"% train_acc)

acc=accuracy_score(y_test,newlabels)
# print("Multilayer Perceptron Classification accuracy for Testing: %f"% acc)
output_f1 = print_metrics(y_test, newlabels, 'SMOTE + Multilayer Perceptron')

x = PrettyTable()
x.field_names = ["Classifier", "Training Score", "Testing Score", "F1-score"]
x.add_row(["SMOTE + Multilayer Perceptron", str(round((train_acc * 100), 2)) + "%", str(round((acc * 100), 2)) + "%" , str(output_f1 *100) + "%"])
print(x)
print ("roc : ",roc_auc_score(y_test, newlabels))


y_true = y_test
y_pred = newlabels
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # Apply 'borderline1' SMOTE + MLP

# In[92]:


pca_new_data = perform_pca(new_data, 2)
X_train, X_test, y_train, y_test = train_test(work,pca_new_data)
pca = PCA(n_components=2)
# kind = ['regular', 'borderline1', 'borderline2', 'svm']
# X_res_train, y_res_train,X_res_test, y_res_test = apply_SMOTE(X_train, y_train,X_test, y_test)
method = SMOTE(kind='borderline1')
X_res_train, y_res_train = method.fit_sample(X_train, y_train)
# X_res_test, y_res_test = method.fit_sample(X_test, y_test)

X_res_test= X_test
y_res_test= y_test  


mlp = MLPClassifier(hidden_layer_sizes=(57,57,57),max_iter=500)
mlp.fit(X_res_train,y_res_train)
newlabels = mlp.predict(X_res_test)

trainlabels=mlp.predict(X_res_train)
train_acc=accuracy_score(y_res_train,trainlabels)
# print("Multilayer perceptron Classification accuracy for Training: %f"% train_acc)

acc=accuracy_score(y_res_test,newlabels)
# print("Multilayer Perceptron Classification accuracy for Testing: %f"% acc)
output_f1 = print_metrics(y_res_test, newlabels, 'SMOTE (borderline1) + Multilayer Perceptron')

x = PrettyTable()
x.field_names = ["Classifier", "Training Score", "Testing Score", "F1-score"]
x.add_row(["SMOTE (borderline1) + Multilayer Perceptron", str(round((train_acc * 100), 2)) + "%", str(round((acc * 100), 2)) + "%" , str(output_f1 *100) + "%"])
print(x)
print ("roc : ",roc_auc_score(y_test, newlabels))


y_true = y_test
y_pred = newlabels
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # Apply 'borderline2' SMOTE + MLP

# In[93]:


pca_new_data = perform_pca(new_data, 2)
X_train, X_test, y_train, y_test = train_test(work,pca_new_data)
# pca = PCA(n_components=5)
# kind = ['regular', 'borderline1', 'borderline2', 'svm']
# X_res_train, y_res_train,X_res_test, y_res_test = apply_SMOTE(X_train, y_train,X_test, y_test)
method = SMOTE(kind='borderline2')
X_res_train, y_res_train = method.fit_sample(X_train, y_train)
# X_res_test, y_res_test = method.fit_sample(X_test, y_test)
X_res_test= X_test
y_res_test= y_test  

mlp = MLPClassifier(hidden_layer_sizes=(57,57,57),max_iter=500)
mlp.fit(X_res_train,y_res_train)
newlabels = mlp.predict(X_res_test)

trainlabels=mlp.predict(X_res_train)
train_acc=accuracy_score(y_res_train,trainlabels)
# print("Multilayer perceptron Classification accuracy for Training: %f"% train_acc)

acc=accuracy_score(y_res_test,newlabels)
# print("Multilayer Perceptron Classification accuracy for Testing: %f"% acc)
output_f1 = print_metrics(y_res_test, newlabels, 'SMOTE (borderline2) + Multilayer Perceptron')

x = PrettyTable()
x.field_names = ["Classifier", "Training Score", "Testing Score", "F1-score"]
x.add_row(["SMOTE(borderline2) + Multilayer Perceptron", str(round((train_acc * 100), 2)) + "%", str(round((acc * 100), 2)) + "%" , str(output_f1 *100) + "%"])
print(x)
print ("roc : ",roc_auc_score(y_test, newlabels))


y_true = y_test
y_pred = newlabels
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # Apply 'svm' SMOTE + MLP

# In[94]:


pca_new_data = perform_pca(new_data, 2)
X_train, X_test, y_train, y_test = train_test(work,pca_new_data)
# pca = PCA(n_components=5)
# kind = ['regular', 'borderline1', 'borderline2', 'svm']
# X_res_train, y_res_train,X_res_test, y_res_test = apply_SMOTE(X_train, y_train,X_test, y_test)
method = SMOTE(kind='svm')
X_res_train, y_res_train = method.fit_sample(X_train, y_train)
# X_res_test, y_res_test = method.fit_sample(X_test, y_test)

X_res_test= X_test
y_res_test= y_test  

mlp = MLPClassifier(hidden_layer_sizes=(57,57,57),max_iter=500)
mlp.fit(X_res_train,y_res_train)
newlabels = mlp.predict(X_res_test)

trainlabels=mlp.predict(X_res_train)
train_acc=accuracy_score(y_res_train,trainlabels)
# print("Multilayer perceptron Classification accuracy for Training: %f"% train_acc)

acc=accuracy_score(y_res_test,newlabels)
# print("Multilayer Perceptron Classification accuracy for Testing: %f"% acc)
output_f1 = print_metrics(y_res_test, newlabels, 'SMOTE (svm) + Multilayer Perceptron')

x = PrettyTable()
x.field_names = ["Classifier", "Training Score", "Testing Score", "F1-score"]
x.add_row(["SMOTE(svm) + Multilayer Perceptron", str(round((train_acc * 100), 2)) + "%", str(round((acc * 100), 2)) + "%" , str(output_f1 *100) + "%"])
print(x)
print ("roc : ",roc_auc_score(y_test, newlabels))


y_true = y_test
y_pred = newlabels
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # Apply 'svm' SMOTE + Naive Bayes

# In[95]:


pca_new_data = perform_pca(new_data, 2)
X_train1, X_test, y_train1, y_test = train_test(work,pca_new_data)


method = SMOTE(kind='svm')
X_train, y_train = method.fit_sample(X_train1, y_train1)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
newlabels=gnb.predict(X_test)


trainlabels=gnb.predict(X_train)
train_acc=accuracy_score(y_train,trainlabels)
# print("PCA + Naive Bayes accuracy for Training: %f"% train_acc)

acc=accuracy_score(y_test,newlabels)
# print("PCA + Naive Bayes Classification accuracy for Testing: ",acc)

output_f1 = print_metrics(y_test, newlabels, 'SMOTE(svm) + naive bayes with PCA')

x = PrettyTable()
x.field_names = ["Classifier", "Training Score", "Testing Score", "F1-score"]
x.add_row(["SMOTE(svm)+naive bayes with PCA", str(round((train_acc * 100), 2)) + "%", str(round((acc * 100), 2)) + "%" , str(output_f1 *100) + "%"])
print(x)
print ("roc : ",roc_auc_score(y_test, newlabels))


y_true = y_test
y_pred = newlabels
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # Apply 'borderline2' SMOTE + Naive Bayes
# 

# In[96]:


pca_new_data = perform_pca(new_data, 2)
X_train1, X_test, y_train1, y_test = train_test(work,pca_new_data)


method = SMOTE(kind='borderline2')
X_train, y_train = method.fit_sample(X_train1, y_train1)
# X_test, y_test = method.fit_sample(X_test1, y_test1)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
newlabels=gnb.predict(X_test)


trainlabels=gnb.predict(X_train)
train_acc=accuracy_score(y_train,trainlabels)
# print("PCA + Naive Bayes accuracy for Training: %f"% train_acc)

acc=accuracy_score(y_test,newlabels)
# print("PCA + Naive Bayes Classification accuracy for Testing: ",acc)

output_f1 = print_metrics(y_test, newlabels, 'SMOTE(borderline2) + naive bayes with PCA')

x = PrettyTable()
x.field_names = ["Classifier", "Training Score", "Testing Score", "F1-score"]
x.add_row(["SMOTE(borderline2)+naive bayes with PCA", str(round((train_acc * 100), 2)) + "%", str(round((acc * 100), 2)) + "%" , str(output_f1 *100) + "%"])
print(x)
print ("roc : ",roc_auc_score(y_test, newlabels))


y_true = y_test
y_pred = newlabels
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # Apply 'borderline1' SMOTE + Naive Bayes

# In[97]:


pca_new_data = perform_pca(new_data, 2)
X_train1, X_test, y_train1, y_test = train_test(work,pca_new_data)


method = SMOTE(kind='borderline1')
X_train, y_train = method.fit_sample(X_train1, y_train1)
# X_test, y_test = method.fit_sample(X_test1, y_test1)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
newlabels=gnb.predict(X_test)


trainlabels=gnb.predict(X_train)
train_acc=accuracy_score(y_train,trainlabels)
# print("PCA + Naive Bayes accuracy for Training: %f"% train_acc)

acc=accuracy_score(y_test,newlabels)
# print("PCA + Naive Bayes Classification accuracy for Testing: ",acc)

output_f1 = print_metrics(y_test, newlabels, 'SMOTE(borderline1) + naive bayes with PCA')

x = PrettyTable()
x.field_names = ["Classifier", "Training Score", "Testing Score", "F1-score"]
x.add_row(["SMOTE(borderline1)+naive bayes with PCA", str(round((train_acc * 100), 2)) + "%", str(round((acc * 100), 2)) + "%" , str(output_f1 *100) + "%"])
print(x)
print ("roc : ",roc_auc_score(y_test, newlabels))


y_true = y_test
y_pred = newlabels
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # Apply 'regular' SMOTE + Naive Bayes

# In[98]:


pca_new_data = perform_pca(new_data, 2)
X_train1, X_test, y_train1, y_test = train_test(work,pca_new_data)


method = SMOTE(kind='regular')
X_train, y_train = method.fit_sample(X_train1, y_train1)
# X_test, y_test = method.fit_sample(X_test1, y_test1)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
newlabels=gnb.predict(X_test)


trainlabels=gnb.predict(X_train)
train_acc=accuracy_score(y_train,trainlabels)
# print("PCA + Naive Bayes accuracy for Training: %f"% train_acc)

acc=accuracy_score(y_test,newlabels)
# print("PCA + Naive Bayes Classification accuracy for Testing: ",acc)

output_f1 = print_metrics(y_test, newlabels, 'SMOTE(regular) + naive bayes with PCA')

x = PrettyTable()
x.field_names = ["Classifier", "Training Score", "Testing Score", "F1-score"]
x.add_row(["SMOTE(regular)+naive bayes with PCA", str(round((train_acc * 100), 2)) + "%", str(round((acc * 100), 2)) + "%" , str(output_f1 *100) + "%"])
print(x)
print ("roc : ",roc_auc_score(y_test, newlabels))


y_true = y_test
y_pred = newlabels
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # PCA + SMOTE + Perceptron

# In[99]:


pca_new_data = perform_pca(new_data, 2)
X_train1, X_test, y_train1, y_test = train_test(work,pca_new_data)


method = SMOTE(kind='regular')
X_train, y_train = method.fit_sample(X_train1, y_train1)
# X_test, y_test = method.fit_sample(X_test1, y_test1)

clf = Perceptron()
clf.fit(X_train,y_train )
newlabels=clf.predict(X_test)

trainlabels=clf.predict(X_train)
train_acc=accuracy_score(y_train,trainlabels)
# print("Perceptron Classification accuracy for Training: %f"% train_acc)
acc=accuracy_score(y_test,newlabels)
# print("Perceptron Classification accuracy for Testing: %f"% acc)
output_f1 = print_metrics(y_test, newlabels, 'Perceptron')

x = PrettyTable()
x.field_names = ["Classifier", "Training Score", "Testing Score", "F1-score"]
x.add_row(["Perceptron", str(round((train_acc * 100), 2)) + "%", str(round((acc * 100), 2)) + "%" , str(output_f1 *100) + "%"])
print(x)
print ("roc : ",roc_auc_score(y_test, newlabels))


y_true = y_test
y_pred = newlabels
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # SVM + gridsearch
# 

# In[48]:


x = PrettyTable()
x.field_names = ["Classifier", "Training Score", "Testing Score", "F1-score"]

X_train, X_test, y_train, y_test = train_test(work,new_data)

parameter_candidates = [
  {'C': [1, 10, 100], 'kernel': ['linear']},
  {'C': [1, 10, 100], 'gamma': [0.001, 0.0001, 2, 10], 'kernel': ['rbf']},
]
clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1);
ci = classifier_info(clf, X_train, X_test, y_train, y_test, "SVM");
ci.classify();
print('[SVM] Best C:',clf.best_estimator_.C) 
print('[SVM] Best Kernel:',clf.best_estimator_.kernel)
print('[SVM] Best Gamma:',clf.best_estimator_.gamma)


# In[49]:


ci.metrics();


# # KNN with gridsearch

# In[50]:


x = PrettyTable()
x.field_names = ["Classifier", "Training Score", "Testing Score", "F1-score"]

X_train, X_test, y_train, y_test = train_test(work,new_data)
k = np.arange(20)+1
parameters = {'n_neighbors': k}
knn = KNeighborsClassifier()
clf = GridSearchCV(knn,parameters,cv=10);
ci = classifier_info(clf, X_train, X_test, y_train, y_test, "kNN");
ci.classify();
ci.print_metrics();


# # KNN + PCA

# In[107]:


pca_new_data = perform_pca(new_data, 2)
X_train, X_test, y_train, y_test = train_test(work,pca_new_data)


neigh = KNeighborsClassifier(n_neighbors=15)
neigh.fit(X_train, y_train)

trainlabels=neigh.predict(X_train)
train_acc=accuracy_score(y_train,trainlabels)
# print("KNN Classification accuracy for Training: %f"% (train_acc))


newlabels = neigh.predict(X_test)
acc=accuracy_score(y_test,newlabels)
# print("KNN Classification accuracy for Testing: %f "% (acc))

output_f1= print_metrics(y_test, newlabels, 'KNN')

x = PrettyTable()
x.field_names = ["Classifier", "Training Score", "Testing Score", "F1-score"]
x.add_row(["KNN", str(round((train_acc * 100), 2)) + "%", str(round((acc * 100), 2)) + "%" , str(output_f1 *100) + "%"])
print(x)
print ("roc : ",roc_auc_score(y_test, newlabels))


y_true = y_test
y_pred = newlabels
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # MLP + PCA 

# In[108]:


pca_new_data = perform_pca(new_data, 2)
X_train, X_test, y_train, y_test = train_test(work,pca_new_data)
    
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
mlp.fit(X_train,y_train)
newlabels = mlp.predict(X_test)

trainlabels=mlp.predict(X_train)
train_acc=accuracy_score(y_train,trainlabels)
# print("Multilayer perceptron Classification accuracy for Training: %f"% train_acc)

acc=accuracy_score(y_test,newlabels)
# print("Multilayer Perceptron Classification accuracy for Testing: %f"% acc)
output_f1 = print_metrics(y_test, newlabels, 'Multilayer Perceptron + PCA')

x = PrettyTable()
x.field_names = ["Classifier", "Training Score", "Testing Score", "F1-score"]
x.add_row(["Multilayer Perceptron + PCA", str(round((train_acc * 100), 2)) + "%", str(round((acc * 100), 2)) + "%" , str(output_f1 *100) + "%"])
print(x)
print ("roc : ",roc_auc_score(y_test, newlabels))


y_true = y_test
y_pred = newlabels
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[109]:


pca_new_data = perform_pca(new_data, 2)
X_train, X_test, y_train, y_test = train_test(work,pca_new_data)
    
    
clf = Perceptron()
clf.fit(X_train,y_train )
newlabels=clf.predict(X_test)

trainlabels=clf.predict(X_train)
train_acc=accuracy_score(y_train,trainlabels)
# print("Perceptron Classification accuracy for Training: %f"% train_acc)
acc=accuracy_score(y_test,newlabels)
# print("Perceptron Classification accuracy for Testing: %f"% acc)
output_f1 = print_metrics(y_test, newlabels, 'Perceptron + PCA')

x = PrettyTable()
x.field_names = ["Classifier", "Training Score", "Testing Score", "F1-score"]
x.add_row(["Perceptron + PCA", str(round((train_acc * 100), 2)) + "%", str(round((acc * 100), 2)) + "%" , str(output_f1 *100) + "%"])
print(x)
print ("roc : ",roc_auc_score(y_test, newlabels))


y_true = y_test
y_pred = newlabels
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# 
# Attribute Information [1]:
# Input variables:
# #bank client data:
# 1 - age (numeric)
# 2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
# 3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
# 4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
# 5 - default: has credit in default? (categorical: 'no','yes','unknown')
# 6 - housing: has housing loan? (categorical: 'no','yes','unknown')
# 7 - loan: has personal loan? (categorical: 'no','yes','unknown')
# #related with the last contact of the current campaign:
# 8 - contact: contact communication type (categorical: 'cellular','telephone') 
# 9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
# 10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
# 
# #other attributes:
# 11 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# 12 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
# 13 - previous: number of contacts performed before this campaign and for this client (numeric)
# 14 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# #social and economic context attributes
# 15 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
# 16 - cons.price.idx: consumer price index - monthly indicator (numeric) 
# 17 - cons.conf.idx: consumer confidence index - monthly indicator (numeric) 
# 18 - euribor3m: euribor 3 month rate - daily indicator (numeric)
# 19 - nr.employed: number of employees - quarterly indicator (numeric)
# 
# Output variable (desired target):
# 20 - y - has the client subscribed a term deposit? (binary: 'yes','no')
# 
