


setwd("C:/Users/Han/Desktop/R practice")
setwd("~/Desktop/ABG")



#### packages
install.packages("ggplot2")
install.packages("ROCR")
install.packages("glmnet")
install.packages("Metrics")
install.packages("DMwR")
install.packages("Rcpp")


library(ggplot2)
library(ROCR)
library(glmnet)
library(Metrics)
#### Input
marketing<- read.csv("marketing.csv")

head(marketing)
summary(marketing)


#### Data Visualization #############################

### Average age for each occupation
ggplot(marketing, aes(job, age)) +
  geom_bar(stat = "summary", fun.y = "mean", color = "black",fill= "grey", width = 0.5) +
  theme_bw() +
    labs( y = "Age", 
        title = "Age Distribution")+
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

ggplot(marketing, aes(job, age, fill = y)) +
  geom_bar(stat = "summary", fun.y = "mean", width = 0.5) +
  theme_bw() +
  labs( y = "Age", 
        title = "Age Distribution")

ggplot(marketing, aes(job, age, fill = y)) +
  geom_bar(stat = "summary", fun.y = "mean", width = 0.5) +
  facet_wrap( ~ marital)
  theme_bw() +
  labs( y = "Age", 
        title = "Age Distribution")
geom_density()


####Data preparation##################################
## Training and Testing
data_y<- marketing[marketing$y == "yes",]
data_n<- marketing[marketing$y == "no", ]

set.seed(1234)
ysub<- sample(nrow(data_y), floor(nrow(data_y)*0.7))
nsub<- sample(nrow(data_n), floor(nrow(data_n)*0.7))

train_yes<- data_y[ysub,]
train_no<- data_n[nsub,]

test_yes<- data_y[-ysub,]
test_no<- data_n[-nsub,]

train<- rbind(train_yes, train_no)
train$y<- ifelse(train$y== "yes", 1, 0)
test<- rbind(test_yes, test_no)
test$y<- ifelse(test$y== "yes", 1, 0)

nrow(marketing)-  nrow(train)- nrow(test)
print(prop.table(table(train$y)))

#### Explore SMOTe 
library(DMwR)

X<- nrow(train_no)
Y<- nrow(train_yes)
perc.over<- ((X-Y)*100/Y)
perc.under<- X*100/(X-Y)

train$y<- as.factor(train$y)
train_bal <- SMOTE(y ~ . , train, perc.over=perc.over, perc.under = perc.under)

print(prop.table(table(train_bal$y)))



################## Model result function
modelperf<- function(ypredict, ytrue, cutoff) {
  library(ROCR)
  ##
  ypredict <- as.numeric(ypredict)
  ytrue<- as.numeric(as.character(ytrue))
  yresult<- ifelse(ypredict > cutoff, 1,0)
  accuracy <- 1 - mean(yresult != ytrue)
  
  ypredict<- as.numeric(ypredict)
  ytrue<- as.numeric(ytrue)
  ROCRpred<- prediction(ypredict, ytrue)
  ROCRperf<- performance(ROCRpred, 'auc')
  ROCRperf
  ROC<- performance(ROCRpred, 'tpr', 'fpr')
  plot(ROC, main = "ROC Curve")
  
  tpr<- sum(yresult == 1 & ytrue == 1)/sum(ytrue == 1)
  tnr<- sum(yresult == 0 & ytrue == 0)/sum(ytrue == 0)
  cm<- matrix(c(tnr, (1- tpr), (1- tnr),tpr), ncol = 2)  
  rownames(cm)<- c("Actual: No", "Actual: Yes")            
  colnames(cm)<- c("Predicted: No", "Predicted: Yes")
  
  result<- list(accuracy, ROCRperf, cm)
  }


### glmnet package local ########################################################
library(glmnet)
## Data preparation

## Use unbalanced data
train_x<-  model.matrix(~age + job+ marital+education+default+balance+housing+loan, train)
train_y<- train$y
test_x<-  model.matrix(~age + job+ marital+education+default+balance+housing+loan, test)
test_y<- test$y  


## Use balanced data
train_x_bal<- model.matrix(~age + job+ marital+education+default+balance+housing+loan, train_bal)
train_y_bal<- train_bal$y
test_x_bal<-  model.matrix(~age + job+ marital+education+default+balance+housing+loan, test)
test_y_bal<- test$y  



### Logistic Regression
glm<- glmnet(x = train_x, y = train_y, family = "binomial", lambda = 0)

lrpred<- predict(glm, newx = test_x, type= "response")

lrperf<- modelperf(lrpred, test_y, 0.5)
lrperf[1]

## Logistic Regression on Balanced data
glm<- glmnet(x = train_x_bal, y = train_y_bal, family = "binomial", lambda = 0)
lrpred_bal<- predict(glm, newx = test_x_bal, type= "response")
lrperf_bal<- modelperf(lrpred_bal, test_y_bal, 0.5)
lrperf_bal[3]
lrperf_bal[1]



### ridge logistic regression
ridge_cv <- cv.glmnet(train_x, train_y, family = "binomial", alpha = 0, type.measure = "auc")
ridge <- glmnet(train_x, train_y, family = "binomial", lambda = ridge_cv$lambda.min)

ridgepred<- predict(ridge_cv, newx = test_x, s = ridge_cv$lambda.min, type = "response")
ridgeperf<- modelperf(ridgepred, test_y, 0.5)

ridgeperf[1]
ridgeperf[3]
## Ridge on balanced data
ridge_cv <- cv.glmnet(train_x_bal, train_y_bal, family = "binomial", alpha = 0, type.measure = "auc")
ridge <- glmnet(train_x_bal, train_y_bal, family = "binomial", lambda = ridge_cv$lambda.min)

ridgepred_bal<- predict(ridge_cv, newx = test_x, s = ridge_cv$lambda.min, type = "response")
ridgeperf_bal<- modelperf(ridgepred_bal, test_y, 0.5)

ridgeperf_bal[2]
ridgeperf_bal[1]



### lasso logistic regression
lasso_cv<- cv.glmnet(train_x, train_y, family = "binomial", alpha = 1, type.measure = "auc")
lasso<- glmnet(train_x, train_y, family = "binomial",lambda = lasso_cv$lambda.min)

lassopred<- predict(lasso, newx = test_x, s = lasso_cv$lambda.min, type = "response")
lassoperf<- modelperf(lassopred, test_y, 0.5)
lassoperf[3]


## Lasso on balanced data
lasso_cv<- cv.glmnet(train_x_bal, train_y_bal, family = "binomial", alpha = 1, type.measure = "auc")
lasso<- glmnet(train_x_bal, train_y_bal, family = "binomial",lambda = lasso_cv$lambda.min)

lassopred_bal<- predict(lasso, newx = test_x, s = lasso_cv$lambda.min)
lassoperf_bal<- modelperf(lassopred_bal, test_y, 0.5)
lassoperf_bal[3]
lassoperf_bal[2]




##Random Forest
install.packages("randomForest")
library(randomForest)

train
head(test)


### unbalanced data
rf<- randomForest(y~., data = train)
rf$confusion
varImpPlot(rf)
rfpred<- predict(rf, test[,1:ncol(test)-1], type = "prob")
rfpred<- as.data.frame(rfpred)
rfperf<- modelperf(rfpred$`1`, test_y, 0.5)
rfperf[1]


### balanced data
rf_bal<- randomForest(y~., data = train_bal)
rf_bal$confusion
varImpPlot(rf)
rfpred_bal<- predict(rf, test[,1:ncol(test)-1], type = "prob")

rfpred<- as.data.frame(rfpred)
rfperf_bal<- modelperf(rfpred$`1`, test_y, 0.5)
rfperf_bal[3]
rfperf_bal[1]


#### SVM
library(e1071)
## SVM on balanced data
train.svm<- train_bal
svm<- svm(y~., data = train.svm)
#test accuracy on test set
svm_test<- test
svm_test$pre<- predict(svm, svm_test, type = "class")
head(svm_test)

svm_pred<- svm_test$pre
yresult<- svm_pred
ytrue<- test$y
tpr<- sum(yresult == 1 & ytrue == 1)/sum(ytrue == 1)
tnr<- sum(yresult == 0 & ytrue == 0)/sum(ytrue == 0)
cm<- matrix(c(tnr, (1- tpr), (1- tnr),tpr), ncol = 2)  
rownames(cm)<- c("Actual: No", "Actual: Yes")            
colnames(cm)<- c("Predicted: No", "Predicted: Yes")
cm




##### H20##################################
install.packages("h2o")
library(h2o)
## Start h2o session
h2o <- h2o.init(nthreads = -1)

## Prepare training, testing, validation set


train_h2o <- as.h2o(train)
test_h2o<- as.h2o(test)
train_bal_h2o<- as.h2o(train_bal)

splits <- h2o.splitFrame(data = test_h2o, 
                         ratios = c(0.5),  #partition data into 70%, 15%, 15% chunks
                         seed = 1234)  #setting a seed will guarantee reproducibility
test_h2o<- splits[[1]]
validate_h2o<- splits[[2]]


train_h2o$y<- as.factor(train_h2o$y)
nrow(test_h2o)
nrow(validate_h2o)
nrow(train_h2o)
#check column index number
colnames(train_h2o)
colnames(test_h2o)


# dependent variable index
y_dep<- match("y",colnames(train_h2o))
# independent variable index
x_ind<- c(1: (y_dep-1))


## Regression family: Logistic, Lasso, ridge

model_logistic<-h2o.glm(x=x_ind,y=y_dep,training_frame=train_h2o,validation_frame=test_h2o, family = "binomial",seed = 1234)
model_logistic
h2o.varimp_plot(model_logistic)

## Lasso regression
model_lasso<-h2o.glm(x=x_ind,y=y_dep,training_frame=train_h2o,validation_frame=test_h2o, family = "binomial",
                     seed = 1234, alpha = 1, lambda_search = TRUE, nlambdas = 100)
model_lasso
h2o.varimp_plot(model_lasso)


## Ridge regression
model_ridge<-h2o.glm(x=x_ind,y=y_dep,training_frame=train_h2o,validation_frame=test_h2o, family = "binomial",
                     seed = 1234, alpha = 0, lambda_search = TRUE, nlambdas = 100)
model_ridge
h2o.varimp_plot(model_ridge)






### Random Forest
model_rf<-h2o.randomForest(x=x_ind,y=y_dep,training_frame=train_h2o,validation_frame=test_h2o,seed = 1234, ntrees = 500)
model_rf

### Naive Bayes
model_bayes<-h2o.naiveBayes(x=x_ind,y=y_dep,training_frame=train_h2o,validation_frame=test_h2o, seed = 1234)
model_bayes
h2o.varimp_plot(model_bayes)



### GBM
model_gbm<-h2o.gbm(x=x_ind,y=y_dep,training_frame=train_h2o,validation_frame=test_h2o, seed = 1234)
model_gbm
h2o.varimp_plot(model_gbm)

## Grid search on GBM model
# Set up parameter
gbm_params <- list(learn_rate = seq(0.01, 0.1, 0.01),
                    max_depth = seq(5, 10, 1),
                    sample_rate = seq(0.5, 1.0, 0.1),
                    col_sample_rate = seq(0.1, 1.0, 0.1))
search_criteria <- list(strategy = "RandomDiscrete", 
                        max_runtime_secs = 60)

gbm_grid <- h2o.grid("gbm", x = x_ind, y = y_dep,
                      grid_id = "gbm_grid",
                      training_frame = train_h2o,
                      validation_frame = test_h2o,
                      seed = 1234,
                      ntrees = 500,
                      hyper_params = gbm_params,
                      search_criteria = search_criteria)

gbm_gridperf <- h2o.getGrid(grid_id = "gbm_grid", 
                             sort_by = "auc", 
                             decreasing = TRUE)

print(gbm_gridperf)

## Select best model from grid search
best_gbm_model_id <- gbm_gridperf@model_ids[[1]]
best_gbm<- h2o.getModel(best_gbm_model_id)

## Test best model on validation set  
best_gbm_perf <- h2o.performance(model = best_gbm, 
                                 newdata = validate_h2o)

best_gbm_perf





#### Deep learning
model_dl<-h2o.deeplearning(x=x_ind,y=y_dep,training_frame=train_h2o,validation_frame=test_h2o, reproducible = T, seed = 123)
model_dl
h2o.varimp_plot(model_dl)



###Grid Search on Deep Learning model
activation_opt <- c("Rectifier", "RectifierWithDropout", "Maxout", "MaxoutWithDropout")
l1_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01, 0.1)
l2_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01, 0.1)
hyper_params <- list(activation = activation_opt,
                     l1 = l1_opt,
                     l2 = l2_opt)
search_criteria <- list(strategy = "RandomDiscrete", 
                        max_runtime_secs = 60)

dl_grid <- h2o.grid("deeplearning", x = x_ind, y = y_dep,
                    grid_id = "dl_grid",
                    training_frame = train_h2o,
                    validation_frame = test_h2o,
                    seed = 1234,
                    hidden = c(10,10),
                    hyper_params = hyper_params,
                    search_criteria = search_criteria)

dl_gridperf <- h2o.getGrid(grid_id = "dl_grid", 
                           sort_by = "auc", 
                           decreasing = TRUE)


print(dl_gridperf)
## Select best Deep learning model on grid search
best_dl_model_id <- dl_gridperf@model_ids[[1]]
best_dl <- h2o.getModel(best_dl_model_id)


best_dl_perf <- h2o.performance(model = best_dl, 
                                newdata = validate_h2o)
h2o.auc(best_dl_perf) 


#### Use final best model to predict on validate test
best_dl_perf
final<- h2o.predict(best_dl, newdata = validate_h2o)
final<- as.data.frame(final)



##########Loss function ###################################

eval_estimate<- function(estimate,truth,loss_FP,loss_FN){
  FP=sum((!truth) & (truth!=estimate))
  FN=sum((truth) & (truth!=estimate))
  TP=sum((truth) & (truth==estimate))
  TN=sum((!truth) & (truth==estimate))
  sens=TP/(TP+FN)
  spec=TN/(TN+FP)
  loss=loss_FP*FP+loss_FN*FN
} 


loss = data.frame()
cuts = data.frame()
for (cut in seq(0, 0.5, by=0.01)){
  res = eval_estimate(ypredprob > cut,ytrue,loss_FP,loss_FN)
  loss = rbind(loss,res)
  cuts = rbind(cuts,cut)
}
cuts_loss = cbind(cuts,loss)
cuts[which.min(loss)]




h2o.shutdown(prompt = TRUE)



