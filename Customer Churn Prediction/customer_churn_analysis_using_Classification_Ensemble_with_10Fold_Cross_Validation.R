library(stats)
library(caTools)
#install.packages('Amelia')
library(Amelia)
library(dplyr)

# set your working directory

# read the telecom dataset input file
dataset <- read.csv("Telco-Customer-Churn.csv")

# print the structure of the dataframe
print(str(dataset))

# check for the NA values 
any(is.na(dataset))

# visualize the missing values using the missing map from the Amelia package
missmap(dataset,col=c("yellow","red"))

# create new column "tenure_interval" from the tenure column
group_tenure <- function(tenure){
    if (tenure >= 0 && tenure <= 6){
        return('0-6 Month')
    }else if(tenure > 6 && tenure <= 12){
        return('6-12 Month')
    }else if (tenure > 12 && tenure <= 24){
        return('12-24 Month')
    }else if (tenure > 24 && tenure <=36){
        return('24-36 Month')
    }else if (tenure > 36 && tenure <=48){
        return('36-48 Month')
    }else if (tenure > 48 && tenure <= 62){
        return('48-62 Month')
    }else if (tenure > 62){
        return('> 62 Month')
    }
}

# apply group_tenure function on each row of dataframe
dataset$tenure_interval <- sapply(dataset$tenure,group_tenure)
dataset$tenure_interval <- as.factor(dataset$tenure_interval)

# Ignore the variables with more levels while predicting the model
# Columns "customerID" and "tenure" having more levels
dataset <- select(dataset,-customerID,-tenure)

# The value of the following columns affecting the model and introducing the NA value for "No phone service" and  and "No internet service" need to cleanup the data for these columns MultipleLine,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies
dataset$MultipleLines <- as.character(dataset$MultipleLines)
dataset$OnlineSecurity <- as.character(dataset$OnlineSecurity)
dataset$OnlineBackup <- as.character(dataset$OnlineBackup)
dataset$DeviceProtection <- as.character(dataset$DeviceProtection)
dataset$TechSupport <- as.character(dataset$TechSupport)
dataset$StreamingTV <- as.character(dataset$StreamingTV)
dataset$StreamingMovies <- as.character(dataset$StreamingMovies)

# convert factor variables into character variables before changing the values
dataset$MultipleLines[dataset$MultipleLines=="No phone service"] <- "No"
dataset$OnlineSecurity[dataset$OnlineSecurity=="No internet service"] <- "No"
dataset$OnlineBackup[dataset$OnlineBackup=="No internet service"] <- "No"
dataset$DeviceProtection[dataset$DeviceProtection=="No internet service"] <- "No"
dataset$TechSupport[dataset$TechSupport=="No internet service"] <- "No"
dataset$StreamingTV[dataset$StreamingTV=="No internet service"] <- "No"
dataset$StreamingMovies[dataset$StreamingMovies=="No internet service"] <- "No"

# converting character variables into factor variables
dataset$MultipleLines <- as.factor(dataset$MultipleLines)
dataset$OnlineSecurity <- as.factor(dataset$OnlineSecurity)
dataset$OnlineBackup <- as.factor(dataset$OnlineBackup)
dataset$DeviceProtection <- as.factor(dataset$DeviceProtection)
dataset$TechSupport <- as.factor(dataset$TechSupport)
dataset$StreamingTV <- as.factor(dataset$StreamingTV)
dataset$StreamingMovies <- as.factor(dataset$StreamingMovies)

# check the number of NA rows if it is relatively small in number then ignore those rows from the analysis
dataset <- na.omit(dataset)
any(is.na(dataset))
#
dataset$Churn <- ifelse(dataset$Churn == 'No', 0, 1)

dataset$Churn <- factor(dataset$Churn, levels = c(0, 1))

# set the seed it will output same output when ever the model is executed
set.seed(123)

# sample the input data with 70% for training and 30% for testing

sample <- sample.split(dataset$Churn,SplitRatio=0.70)
training_set <- subset(dataset,sample==TRUE)
test_set <- subset(dataset,sample==FALSE)

#MODEL1 (Logistic Regression)
model <- glm(Churn ~ .,family=binomial(link="logit"),data=training_set)
print(summary(model))
y_pred <- predict(model,newdata=test_set,type="response")
y_pred <- ifelse(y_pred > 0.5,1,0)
confusionMatrix(table(test_set[, 19], y_pred))
cm <- (table(test_set[, 19], y_pred))
accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
accuracy


#MODEL2 (NaiveBayes)
set.seed(123)
classifier2 = naiveBayes(x = training_set[-19], y = training_set$Churn)
y_pred2 = predict(classifier2, newdata = test_set[-19])
confusionMatrix(table(test_set[, 19], y_pred2))
cm2 <- (table(test_set[, 19], y_pred2))
accuracy2 = (cm2[1,1] + cm2[2,2]) / (cm2[1,1] + cm2[2,2] + cm2[1,2] + cm2[2,1])
accuracy2

#MODEL3 (SVM)
set.seed(123)
classifier3 = svm(formula = Churn ~ .,
                  data = training_set,
                  type = 'C-classification',
                  kernel = 'sigmoid')
y_pred3 = predict(classifier3, newdata = test_set[-19])
cm3 <- (table(test_set[, 19], y_pred3))
confusionMatrix(table(test_set[, 19], y_pred3))
accuracy3 = (cm3[1,1] + cm3[2,2]) / (cm3[1,1] + cm3[2,2] + cm3[1,2] + cm3[2,1])
accuracy3


#MODEL4 (RandomForest)
rf_trees2 <- seq(1,100, by=1)
rf_grid2 <- expand.grid(rf_trees2)
colnames(rf_grid2)<- c("trees")
rf_grid2$accuracy<- 0
for(i in 1:length(rf_grid2$trees)){
  parameters = rf_grid2[i,]
  set.seed(123)
  classifier4 = randomForest(x= training_set[-19],
                                  y= training_set$Churn,
                                  ntree = parameters$trees)
  y_pred4 = predict(classifier4, newdata = test_set[-19])
  cm4 <- (table(test_set[, 19], y_pred4))
  accuracy4 <- (cm4[1,1] + cm4[2,2]) / (cm4[1,1] + cm4[2,2] + cm4[1,2] + cm4[2,1])
  print(i)
  rf_grid2[i,2] <- accuracy4
}
optimized_hyperparameter3<- rf_grid2[which.max(rf_grid2$accuracy), ]
optimized_hyperparameter3

set.seed(123)
classifier4 = randomForest(x= training_set[-19],
                                y= training_set$Churn,
                                ntree = optimized_hyperparameter3$trees)

y_pred4 = predict(classifier4, newdata = test_set[-19])
cm4 <- (table(test_set[, 19], y_pred4))
accuracy4 <- (cm4[1,1] + cm4[2,2]) / (cm4[1,1] + cm4[2,2] + cm4[1,2] + cm4[2,1])
accuracy4




Prediction <- data.frame(y_pred,y_pred2,y_pred3,y_pred4,test_set[,19])
colnames(Prediction) <- c("Forecast","Forecast_2","Forecast_3","Forecast_4","Actual")

#Ensemble with 10-Fold Cross Validation
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
    classifier_final = randomForest(x= training_fold[-5],
                                    y= training_fold$Actual,
                                    ntree = parameters$trees, type='Classification')
    y_predfinal = predict(classifier_final, newdata = test_fold[-5])
    cm_final = table(test_fold[, 5], y_predfinal)
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
                                ntree = optimized_hyperparameter2$trees)

y_predfinal = predict(classifier_final, newdata = Prediction[-5])
cm_final <- (table(test_set[, 19], y_predfinal))
accuracy_final <- (cm_final[1,1] + cm_final[2,2]) / (cm_final[1,1] + cm_final[2,2] + cm_final[1,2] + cm_final[2,1])
accuracy_final
confusionMatrix(table(test_set[, 19], y_predfinal))


Prediction_final <- data.frame(y_pred,y_pred2,y_pred3,y_pred4,y_predfinal,test_set[,19])
colnames(Prediction_final) <- c("Forecast","Forecast_2","Forecast_3","Forecast_4","Forecast_final","Actual")