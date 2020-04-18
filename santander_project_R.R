rm(list = ls())

#Setting the working Directory first
setwd("C:/Users/acer/Desktop/edWisor Project1")

#re-check the working directory
getwd()

#Loading important Libraries
library(caret) # For various perfomance parameters
library(caTools) # FOr splitting the datasett
library(dplyr) # For balancing the target class of dataset
library(corrplot) #For plotting the correlation heatmap
library(ggplot2) #For visualization
library(Matrix)
library(pROC) #For calculating the ROC-AUC Score
library(C50) #For using Decision Tree Algorithm
library(randomForest) # For using Random Forest Algorithm
library(e1071) # for using Naive Bayes Algorithm

#Load the train & test dataset into the R environment.
df_train = read.csv("train.csv", header = TRUE)
df_test = read.csv("test.csv",header = TRUE)


####################### Exploratory Data Analysis (EDA) #################################

###### 1. EXAMINING THE DATASET

View(df_train)
View(df_test)

# After viewing both the train and test data:

# 1. Both the dataset are having 200000 observations. The train set has 202 whereas test set has 201 features/variables/predictors

# 2. Note that the feature "ID_code" in both the dataset is just representing index of observations and it will not contribute 
#    anything in our model so we'll just drop this variable 

df_train = df_train[,-1]
df_test = df_test[,-1]

# See the statistical parameters of the rest of the predictors/variables 

summary(df_train)
summary(df_test)

str(df_train)
str(df_test)
#After viewing the dataset summary we can conclude all the indipendent variables are numeric in nature and are having numerical values in a narrow range.



#Let's check the distribution of target class in the train dataset
table(df_train$target)
#Visualization
hist(df_train$target,col = "green", main = paste("Imbalanced Target Class Distribution"))


#   We can clearly see from the plot that the target class is imbalanced and majority and minority classes have 1,79,902 & 20,098
#   observations i.e. having a ratio of approx 9:1

#  This imbalance in target class will create a bias towards the Majority class in the prediction so we need 
#  to keep this in mind before training our model.


###### 2.DATA CLEANING


##  Missing Value Analysis


#Check for the missing value
sum(is.na(df_train)) 
sum(is.na(df_test))

#result shows that no missing values are there in both train and test dataset



### creating dependent and independent arrays
X = df_train[,-1]
y = df_train[,1]


####### 3. FEATURE SELECTION

### Removal of constant features

names(df_train[,sapply(df_train, function(v) var(v, na.rm = TRUE)== 0)])

df_train[,sapply(df_train, function(v) var(v, na.rm = TRUE)== 0)]



### Checking for the Correlation between all the variable as all are of numeric type with the help of a heatmap

corrplot(cor(X))
corrplot(cor(df_test))
# we can see from the Correlation plot (heatmap) that there are no correlated features in the dataset



#########   Spliting the dataset in train and validation set #############


set.seed(121)

split = sample.split(df_train$target,SplitRatio = 0.8)
  
training_Set = subset(df_train, split==TRUE)
Validation_set = subset(df_train,split==FALSE)

print(dim(training_Set))
print(dim(Validation_set))

X_train = training_Set[-1]
y_train = training_Set[1]

X_val = Validation_set[-1]
y_val = Validation_set[1]


######## Handling the Imbalanced target class #############


#    Now as our dataset is huge i.e. it has 2lacs observation we can perform undersampling
#    method to balance as well as sample our data set as it will take huge amount
#    of time and computaion power to perform operations on 2Lacs observations, therefore we will use the Random Under 
#    Sampling technique to sample as well as balance our dataset

#Sampling the training dataset 

Majority_class = subset(training_Set,training_Set$target==0)

Minority_class = subset(training_Set,training_Set$target==1)

dim(Majority_class)[1]

#Undersampling the Majority Class
set.seed(123)
training_Set_Und_samp<- training_Set %>%
  group_by(target) %>%
  sample_n(dim(Minority_class)[1])


dim(training_Set_Und_samp)

table(training_Set_Und_samp$target)


X_train_res = training_Set_Und_samp[-1]
y_train_res = training_Set_Und_samp[1]





#############################      Model Creation   ################

#Let's first define a function that returns the variaous important performance metrics


Performance_metrics <- function(y_true,y_pred) {
  cm = as.matrix(table(y_true,y_pred))
  Accuracy = sum(diag(cm))/sum(cm)
  Precision_m = cm[1,1]/(cm[1,1]+cm[1,2])
  Recall_m = cm[2,2]/(cm[2,2]+cm[2,1])
  ROC_AUC_Score = roc(y_actual,as.numeric(y_pred))
  auc_score = auc(ROC_AUC_Score)
  FNR = cm[2,1]/(cm[2,1]+cm[2,2])
  FPR = cm[1,2]/(cm[1,2]+cm[1,1])
  
  paste0("Accuracy: ",Accuracy, ",  Precision: ",Precision_m,",  Recall : ",Recall_m,",  ROC-AUC Score: ", auc_score,  ",  FPR :", FPR, ",  FNR: ", FNR)
  
}

#################### 1. LOGISTIC REGRESSION  #################
# We will start the creation of model with our basic linear model Logistic Regression


clf_LR = glm(target~.,data=training_Set_Und_samp, family = "binomial")

summary(clf_LR)
prob_pred = predict(clf_LR, type = "response",newdata = Validation_set[-1])

y_pred_LR = ifelse(prob_pred > 0.5,"1","0")

y_actual = Matrix::t(y_val)

#Confusion Matrix
cm_LR = as.matrix(table(y_actual,y_pred_LR))
cm_LR

#Evaluating Performance Metrics
Performance_metrics(y_actual,y_pred_LR)

# We get an accuracy of 77.79 % & a recall score of 77.51% with the Logistic regression model
# Let's evaluate some more important meterics for our result





##################  2.Decision Tree Algorithms ################

# As our Logistic regression model doesn't provide that great result let's try decision tree method now



training_Set_Und_samp$target = as.factor(training_Set_Und_samp$target)

clf_DT = C5.0(x = training_Set_Und_samp[-1],y = training_Set_Und_samp$target)

y_pred_DT = predict(clf_DT, type = "class", newdata = X_val)

cm_DT = as.matrix(table(y_actual,y_pred_DT))
cm_DT

Performance_metrics(y_actual,y_pred_DT)


#Showing the tree classification
plot(clf_DT)
text(clf_DT)




############### 3. Random Forest Algorithm #######################


clf_RF = randomForest(target ~.,data = training_Set_Und_samp, importance = TRUE,ntree = 500)

y_pred_RF = predict(clf_RF, X_val, type = "class")


cm_RF = as.matrix(table(y_actual,y_pred_RF))
cm_RF

Performance_metrics(y_actual,y_pred_RF)



#Showing the classification
plot(clf_RF)
text(clf_RF)







################## 4. Naive Bayes Algorithm ################


clf_NB = naiveBayes(target~.,data = training_Set_Und_samp)

y_pred_NB = predict(clf_NB, type = "class", newdata = X_val)

cm_NB = as.matrix(table(y_actual,y_pred_NB))
cm_NB

Performance_metrics(y_actual,y_pred_NB)







################################################ Model Selection ##############################################



#Based on the performance metrics of various models we will select the Naive Bayes as along with providing 
# great accuracy it also has the higher ROC-AUC Score and high Recall value.

# Now let's predict the test dataset results and store the file


y_pred_test = predict(clf_NB,newdata = df_test,type = "class")


# Adding this predicted vector of target class into the df_test dataframe and store it in the working directory
# location by the name of Submission_R.csv

df_test$predicted_target_Class = y_pred_test

head(df_test)

write.csv(df_test,"C:/Users/acer/Desktop/edWisor Project1/Submission_R.csv",row.names=FALSE)

