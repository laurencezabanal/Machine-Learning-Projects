#Import file
customer_loan_details <- read.csv("customer_loan_details.csv", sep = ",")

# Print the structure of the dataframe
str(customer_loan_details)
head(customer_loan_details)

# Check for the NA values
any(is.na(customer_loan_details))

# Calculating DTI
customer_loan_details$dti <- (customer_loan_details$debts/customer_loan_details$income)*100

# Create loan_decision_status variable which is our target variable to use for loan prediction
customer_loan_details$loan_decision_status <- ifelse(customer_loan_details$loan_decision_type == 'Denied', 0, 1)

# Encoding the target variable as factor
customer_loan_details$loan_decision_status <- factor(customer_loan_details$loan_decision_status, levels = c(0, 1))

#Selecting the required fields for prediction
customer_loan_refined <- customer_loan_details[,c(3,4,6:8,11,13:14)]
head(customer_loan_refined)

# Encoding the categorical variable as factors
customer_loan_refined$gender <- as.numeric(factor(customer_loan_refined$gender,
                                                  levels = c('Male','Female'),
                                                  labels = c(1,2)))

customer_loan_refined$marital_status <- as.numeric(factor(customer_loan_refined$marital_status,
                                                          levels = c('Divorced','Married','Single'),
                                                          labels = c(1,2,3)))

customer_loan_refined$occupation <- as.numeric(factor(customer_loan_refined$occupation,
                                                      levels = c('Accout','Business','IT','Manager','NYPD'),
                                                      labels = c(1,2,3,4,5)))

customer_loan_refined$loan_type <- as.numeric(factor(customer_loan_refined$loan_type,
                                                     levels = c('Auto','Credit','Home','Personal'),
                                                     labels = c(1,2,3,4)))

head(customer_loan_refined)

# Splitting the customer_loan_refined dataset into training and test sets
library(caTools)
set.seed(123)
split = sample.split(customer_loan_refined$loan_decision_status, SplitRatio = 0.70)
training_set = subset(customer_loan_refined, split == TRUE)
test_set = subset(customer_loan_refined, split == FALSE)

#Applying Feature Scaling
training_set[-8] = scale(training_set[-8])
test_set[-8] = scale(test_set[-8])

head(training_set)

# Applying Dimensionality reduction using PCA to training and test sets
#install.packages("caret")
library(caret)
pca = preProcess(x = training_set[-8], method = 'pca', pcaComp = 2)
training_set_pca = predict(pca, training_set)
training_set_pca = training_set_pca[c(2, 3, 1)]
test_set_pca = predict(pca, test_set)
test_set_pca = test_set_pca[c(2, 3, 1)]
head(test_set_pca)

# Appling Naive Bayes classification model to predict the loan
# install.packages("e1071")
library(e1071)
#install.packages('xgboost')
library(xgboost)
library(randomForest)
library(class)

#MODEL1 (NaiveBayes)
set.seed(123)
classifier = naiveBayes(x = training_set_pca[-3], y = training_set_pca$loan_decision_status)
y_pred = predict(classifier, newdata = test_set_pca[-3])
confusionMatrix(table(test_set_pca[, 3], y_pred))
cm <- (table(test_set_pca[, 3], y_pred))
accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
accuracy

#MODEL2 (SVM)
set.seed(123)
classifier2 = svm(formula = loan_decision_status ~ .,
                  data = training_set_pca,
                  type = 'C-classification',
                  kernel = 'sigmoid')
y_pred2 = predict(classifier2, newdata = test_set_pca[-3])
cm2 <- (table(test_set_pca[, 3], y_pred2))
confusionMatrix(table(test_set_pca[, 3], y_pred2))
accuracy2 = (cm2[1,1] + cm2[2,2]) / (cm2[1,1] + cm2[2,2] + cm2[1,2] + cm2[2,1])
accuracy2

#MODEL3 (KNN)
knn_k <- seq(1,50, by=1)
knn_grid <- expand.grid(knn_k)
colnames(knn_grid)<- c("k")
knn_grid$accuracy<- 0
for(i in 1:length(knn_grid$k)){
  parameters = knn_grid[i,]
  set.seed(123)
  y_pred3 = knn(train = training_set_pca[-3],
                test = test_set_pca[,-3],
                cl = training_set_pca[,3],
                k = parameters$k)
  cm3 <- (table(test_set_pca[, 3], y_pred3))
  accuracy3 = (cm3[1,1] + cm3[2,2]) / (cm3[1,1] + cm3[2,2] + cm3[1,2] + cm3[2,1])
  print(i)
  knn_grid[i,2] <- accuracy3
}
optimized_hyperparameter<- knn_grid[which.max(knn_grid$accuracy), ]
optimized_hyperparameter

set.seed(123)
y_pred3 = knn(train = training_set_pca[-3],
              test = test_set_pca[,-3],
              cl = training_set_pca[,3],
              k = optimized_hyperparameter$k)
cm3 <- (table(test_set_pca[, 3], y_pred3))
accuracy3 = (cm3[1,1] + cm3[2,2]) / (cm3[1,1] + cm3[2,2] + cm3[1,2] + cm3[2,1])
accuracy3

#MODEL4 (Random Forest)
rf_trees2 <- seq(1,100, by=1)
rf_grid2 <- expand.grid(rf_trees2)
colnames(rf_grid2)<- c("trees")
rf_grid2$accuracy<- 0
for(i in 1:length(rf_grid2$trees)){
  parameters = rf_grid2[i,]
  set.seed(123)
  classifier4 = randomForest(x= training_set_pca[-3],
                                  y= training_set_pca$loan_decision_status,
                                  ntree = parameters$trees,type='classification')
  y_pred4 = predict(classifier4, newdata = test_set_pca[-3])
  cm4 <- (table(test_set_pca[, 3], y_pred4))
  accuracy4 <- (cm4[1,1] + cm4[2,2]) / (cm4[1,1] + cm4[2,2] + cm4[1,2] + cm4[2,1])
  print(i)
  rf_grid2[i,2] <- accuracy4
}
optimized_hyperparameter3<- rf_grid2[which.max(rf_grid2$accuracy), ]
optimized_hyperparameter3

set.seed(123)
classifier4 = randomForest(x= training_set_pca[-3],
                           y= training_set_pca$loan_decision_status,
                           ntree = optimized_hyperparameter3$trees,type='classification')
y_pred4 = predict(classifier4, newdata = test_set_pca[-3])
cm4 <- (table(test_set_pca[, 3], y_pred4))
accuracy4 <- (cm4[1,1] + cm4[2,2]) / (cm4[1,1] + cm4[2,2] + cm4[1,2] + cm4[2,1])
accuracy4

#MODEL5 (Logistic Regression)
model <- glm(loan_decision_status ~ .,family=binomial(link="logit"),data=training_set_pca)
#print(summary(model))
y_pred5 <- predict(model,newdata=test_set_pca,type="response")
y_pred5 <- ifelse(y_pred5 > 0.5,1,0)
#confusionMatrix(table(test_set[, 19], y_pred))
cm5 <- (table(test_set_pca[, 3], y_pred5))
accuracy5 = (cm5[1,1] + cm5[2,2]) / (cm5[1,1] + cm5[2,2] + cm5[1,2] + cm5[2,1])
accuracy5


# Predicting the Test set results

#y_pred3 = predict(classifier3, newdata = test_set_pca[-3])
#y_pred3 = predict(classifier3, newdata = as.matrix(test_set_pca[-3]))

Prediction <- data.frame(y_pred,y_pred2,y_pred3,y_pred4,y_pred5,test_set_pca[,3])
colnames(Prediction) <- c("Forecast","Forecast_2","Forecast_3","Forecast_4","Forecast_5","Actual")

#Prediction <- data.frame(y_pred,y_pred2,y_pred3,test_set_pca[,3])
#colnames(Prediction) <- c("Forecast","Forecast_2","Forecast_3","Actual")


#Ensemble with Cross Validation
library(caret)

rf_trees <- seq(1,100, by=1)
rf_grid <- expand.grid(rf_trees)
colnames(rf_grid)<- c("trees")
rf_grid$accuracy<- 0
for(i in 1:length(rf_grid$trees)){
  parameters = rf_grid[i,]
  set.seed(123)
  folds=createFolds(Prediction$Actual,k=10) 
  cv = lapply(folds, function(x){
    training_fold = Prediction[-x,]
    test_fold = Prediction[x,]
    set.seed(123)
    classifier_final = randomForest(x= training_fold[-6],
                                    y= training_fold$Actual,
                                    ntree = parameters$trees, type='Classification')
    y_predfinal = predict(classifier_final, newdata = test_fold[-6])
    cm_final = table(test_fold[, 6], y_predfinal)
    accuracy_final <- (cm_final[1,1] + cm_final[2,2]) / (cm_final[1,1] + cm_final[2,2] + cm_final[1,2] + cm_final[2,1])
    return(accuracy_final)
  })
  accuracy_final <- mean(as.numeric(cv))
  print(i)
  rf_grid[i,2] <- accuracy_final
}
optimized_hyperparameter2<- rf_grid[which.max(rf_grid$accuracy), ]
optimized_hyperparameter2

set.seed(123)
classifier_final = randomForest(x= Prediction[-5],
                                y= Prediction$Actual,
                                ntree = optimized_hyperparameter2$trees, type='Classification')

y_predfinal = predict(classifier_final, newdata = Prediction[-5])
cm_final <- (table(test_set_pca[, 3], y_predfinal))
accuracy_final <- (cm_final[1,1] + cm_final[2,2]) / (cm_final[1,1] + cm_final[2,2] + cm_final[1,2] + cm_final[2,1])
accuracy_final
confusionMatrix(table(test_set_pca[, 3], y_predfinal))

# Visualising the Test set results
#install.packages("ElemStatLearn")