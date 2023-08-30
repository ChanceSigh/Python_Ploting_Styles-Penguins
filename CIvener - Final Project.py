# # Buisness Understanding

# #### We are being asked to find penguins of a specific species and to do so we will use two models k nearest neigbors and desision trees

# In[23]:


#import required tools
import pandas as pd
from sklearn import tree
from sklearn import metrics
from sklearn import model_selection as skms
from sklearn import neighbors
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from jupyterthemes import jtplot
jtplot.style()


# # Data Understanding

# #### The data contains seven columns. Species which is contained in a object data type. Island which is the location, which is also contained in a object data type. Bill length which is the length of the pengins bill is a float. Bill depth which is the depth of the bill also measued as a float. Flipper length is then measured as a float. Then body mass which is measured in grams is a float. Finally the sex of the penguin is measured as a object. We will have to convert all but species into usable floats for the following models

# In[24]:


#read the original data
penguin = pd.read_csv('https://raw.githubusercontent.com/ChanceSigh/Python_Ploting_Styles-Penguins/main/penguins_clean.csv', na_values=['Unknown'])
penguin.info()


# # Data Preperation - General

# In[25]:


#drop na
penguin = penguin.dropna()


# In[26]:


#convert sex to numeric values
penguin['sex'] = penguin['sex'].map({'Female':0, 'Male':1})


# In[27]:


#one hot encoding to convert island
penguin_hot = pd.get_dummies(penguin, columns=['island'], drop_first=True)


# In[28]:


#save data to convert uint8 to a usable data type... have not learned in class of any other way to convert
penguin_hot.to_csv('../working/penguin_hot_encoding.csv', index=False)
penguin_hot1 = pd.read_csv('../working/penguin_hot_encoding.csv')

#to change data type... not used in class but is in the jupyter help section
penguin_hot1['island_Dream'] = penguin_hot1.island_Dream.astype(float)
penguin_hot1['island_Torgersen'] = penguin_hot1.island_Torgersen.astype(float)
penguin_hot1['sex'] = penguin_hot1.sex.astype(float)


# # K-NN - Data Prep

# In[29]:


#seperate target and feturs and model base values
TEST_SIZE = 0.2
RANDOM_STATE = 120
target = penguin_hot1['species']
features = penguin_hot1.drop(columns=['species'])

#split data
tts = skms.train_test_split(features, target,
                           test_size=TEST_SIZE, random_state=RANDOM_STATE)

#save the split data
(train_ftrs, test_ftrs, train_target, test_target) = tts


# In[30]:


#K-NN standardizing
stdsc = StandardScaler()
train_std = stdsc.fit_transform(train_ftrs)
test_std = stdsc.transform(test_ftrs)


# # K-NN -- Modeling/Data Exploration

# In[31]:


#fit preped data for prediction
K = 5
knn = neighbors.KNeighborsClassifier(n_neighbors=K)
model = knn.fit(train_std, train_target)
preds = model.predict(test_std)


# In[32]:


#accuracy score of K-NN
metrics.accuracy_score(test_target,preds)


# In[33]:


#set up preds for confusion matrix
preds = model.predict(test_ftrs)

#create confusion matrix
cm = metrics.confusion_matrix(test_target,preds)
cm_disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                        display_labels=model.classes_)
cm_disp.plot()


# ## K-NN Model and Data Explanation
# ###### The accuracy shown is 100 percent accurate. It made 25 true positive predictions and had 42 False negatives. Though it has a high accuracy I would not trust the predictions since False negatives are so high within it.

# # Desision Tree - Data Prep

# In[34]:


#create constraits
LABEL_NAME = 'species'
df = penguin_hot1

#seperate target and featurs
target = df[LABEL_NAME]
features = df.drop(columns=[LABEL_NAME])

#set size and random state with split
tts = skms.train_test_split(features, target, 
                            test_size=0.2, random_state=120)

#save split data
(train_ftrs, test_ftrs, train_target, test_target) = tts


# # Desision Tree - Model Building/Data Exploration

# In[35]:


#fit for tree model
penguin_tree = tree.DecisionTreeClassifier(criterion='gini', max_depth=3)
model = penguin_tree.fit(train_ftrs, train_target)
#accuracy score
model.score(test_ftrs, test_target)


# In[36]:


#plot decision tree
plt.figure(figsize=(25,12))
tree.plot_tree(model,
               feature_names=test_ftrs.columns,
               class_names=model.classes_,
               filled=True,
               fontsize=12)
plt.show()


# In[37]:


#set preds for confusion matrix
preds = model.predict(test_ftrs)

#confusion matrix
cm = metrics.confusion_matrix(test_target,preds)
cm_disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                        display_labels=model.classes_)
cm_disp.plot()


# ## Desision Tree Model and Data Explanation
# ###### The model is over 98 percent accurate. The tree shows a direct path for each species based on the traits. Meaning certain traits only belong to certain species, making the accuracy better. The confusion matrix showed only one false negative. With that I would trust this model more than I would others.

# # General Model Building/Data Exploration

# In[38]:


#scatter plot for bill data
penguin.plot.scatter(x='bill_length_mm',
                 y='bill_depth_mm',
                 grid=True,
                 marker='.',
                 c='red',)


# In[39]:


#scatter plot for sex based on size data
penguin.plot.scatter(x='flipper_length_mm',
                y='body_mass_g',
                c='sex',
                colormap='Set3')


# ## General Model and Data explanition
# ###### The main purpose of this was to check corrilation between the species. Even though it is not directly stated it should be assumed that when the values are grouped as they are in the plots that it means those groupings are the variation in species. Being that identical species should have simular traits. It would also show why the other models done might have an error. The fact that when I seperated them by sex we could see that there are multiple groupings showing that the gender is a defining trait causing a abnormal gap compared to the other data used. We can also see that there is a area between both major groupings in the plots that has outliers. This shows that pengins can have abnormal characteristics that would make it meet the requirments of another species. Being a possible cause for why the other models had some error.

# # Evaluation
# 
# ###### The models both gave quite high accuracy scores. Though the k nearest neighbors did report a multitude of incorect results. The decision tree on the other hand had only one incorrect prediction. I belive this is due to the fact that the decision tree focuses on decerning features and sperates them based on that, while the k nearest neighbors standardizes the data in a way that looks for the closest set of data to each group. For this data set one of the species had traits that were far to simular to the others, causing the prediction to favor that specific one. That is why it predicted the same species for its results. While on the other hand the decision tree would split the species by their data into different groups. By doing so the species that had a mixture of traits was correctly identified. This alowed it to make predictions unimpared by the squew in the traits.
# 
# ###### I would perfer to use the decision tree in this senario. Being that it is more favorable to give a true possitive prediction. Though k nearest neigbors does have its benifits in certain senarios I do not belive this is it. If given finacial data to make a prediction I would beleive that k nearest neigbors would be benificial. Being that there are generally less outliers and missinterperted data within finance. When if comes to pure seperation of groups by common values I would have to perfer decision trees. They seperate groups finding the most distict traits every seperation to find the most influential components. Making it great when the data is otherwise ambigous.
