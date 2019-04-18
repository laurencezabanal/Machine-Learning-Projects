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

#MODEL1
set.seed(123)
classifier = naiveBayes(x = training_set_pca[-3], y = training_set_pca$loan_decision_status)
y_pred = predict(classifier, newdata = test_set_pca[-3])
confusionMatrix(table(test_set_pca[, 3], y_pred))
cm <- (table(test_set_pca[, 3], y_pred))
accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
accuracy

#MODEL2
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

#MODEL3
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


# Predicting the Test set results

#y_pred3 = predict(classifier3, newdata = test_set_pca[-3])
#y_pred3 = predict(classifier3, newdata = as.matrix(test_set_pca[-3]))

Prediction <- data.frame(y_pred,y_pred2,y_pred3,test_set_pca[,3])
colnames(Prediction) <- c("Forecast","Forecast_2","Forecast_3","Actual")

#Prediction <- data.frame(y_pred,y_pred2,y_pred3,test_set_pca[,3])
#colnames(Prediction) <- c("Forecast","Forecast_2","Forecast_3","Actual")


#Ensemble
set.seed(123)
classifier_final = randomForest(x= Prediction[-4],
                           y= Prediction$Actual,
                           ntree = 100)

y_predfinal = predict(classifier_final, newdata = Prediction[-4])
cm_final <- (table(test_set_pca[, 3], y_predfinal))
accuracy_final <- (cm_final[1,1] + cm_final[2,2]) / (cm_final[1,1] + cm_final[2,2] + cm_final[1,2] + cm_final[2,1])
accuracy_final
confusionMatrix(table(test_set_pca[, 3], y_predfinal))


# Visualising the Test set results
#install.packages("ElemStatLearn")
library(ElemStatLearn)
set = test_set_pca

# Built the grid using X1, X2 by taking min-1 and max+1 values from test set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')

# Predict the test set observations 
y_grid = predict(classifier_final, newdata = grid_set[-3])

# Plot the graph using actual observations from test set and predicted results
plot(set[, -3], main = 'Naive Bayes (Test set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))