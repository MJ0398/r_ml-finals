library(tidyverse)
library(magrittr)
library(outliers)
library(caret)
library(doParallel)

seed = 323

#1. Data Preprocessing ----
data.raw = read.csv("intro_extravert.csv")
 #1250 obs, 94 vars

#1.1 Data overview
str(data.raw) #all int
data.raw %>% view()

#1.2 check NA / Impute NA
data.raw %>% anyNA #no NAs

#1.2.A. Setting NAs: converting IE, gender, engnat = 0 to NA
data.raw %<>% 
  mutate(IE = replace(IE, IE == 0, NA))

data.raw %<>% 
  mutate(gender = replace(gender, gender == 0, NA))

data.raw %<>% 
  mutate(engnat = replace(engnat, engnat == 0, NA))

#Check NAs
data.raw %>% map_dbl(~sum(is.na(.))) #NA is set

#Dropping the NAs
data.raw %<>% na.omit()
data.raw %>% anyNA #no NA

#1.2.B. Setting NAs: convert outliers to Nas, impute NAs
outlier(data.raw)
data.raw %>% Hmisc::hist.data.frame() #check Q17
data.raw$Q17A #-77 seems suspicious, impute it to median
data.raw %<>% mutate(Q17A = replace(Q17A, Q17A == -77, NA))

#check NA
data.raw %>% anyNA
data.raw %>% map_dbl(~sum(is.na(.))) #Q17A, 131

#impute NA
set.seed(seed)
data.raw %<>% 
  # preProcess(method = "knnImpute") %>% #not working!!
  preProcess(method = "medianImpute") %>%
  # preProcess(method = "bagImpute") %>% 
  predict(data.raw) %T>% print

#check imputation
data.raw %>% anyNA
data.raw %>% summary()
data.raw$Q17A

#1.3 changing data types to relevant data
data.raw %>% str()
#IE, gender, engnat to factors
data.raw %<>% mutate_at(vars(IE: engnat), as.factor)
#checking
str(data.raw)
data.raw$IE %>% levels()
data.raw$gender %>% levels()
data.raw$engnat %>% levels()

#1.4 Feature Selection
#There is no particular feature to be removed. However, if we want to classify the respondants
#into introverts or extroverts, level 3 (neither), should be removed
#from the target variable. Also, naming should be changed to Y, N
data.raw$IE %>% table()
dataset = data.raw %>% mutate(IE = replace(IE, IE == "3", NA))
dataset %<>% drop_na()
#check, making 2 leveled factor
dataset$IE %>% table()
dataset$IE %<>% as.numeric
dataset$IE %<>% as.factor
dataset$IE %>% levels()
dataset$IE %>% class()
dataset %>% anyNA
str(dataset)
dataset %>% view()
levels(dataset$IE)[levels(dataset$IE)=="1"] <- "I"
levels(dataset$IE)[levels(dataset$IE)=="2"] <- "E"

#2. Splitting Data & Configurations  ----

#2.1 Splitting Data
target.label = "IE"
set.seed(seed)
train.index = createDataPartition(dataset[[target.label]], p = 0.9, list = F)
trainset = dataset[train.index,]
testset = dataset[-train.index,]

#checking training & testing data formation
trainset[[target.label]] %>% plot()
testset[[target.label]] %>% plot() #similar enough

#2.2 Target Selection & Formula Creation
target = trainset[[target.label]]
features.label = trainset %>% select(-target.label) %>% names() %T>% print()
features = trainset %>% select(features.label) %>% as.data.frame()

formula = features %>%
  names() %>% 
  paste(., collapse = " + ") %>% 
  paste(target.label, "~ ", .) %>% 
  as.formula(env = .GlobalEnv) %T>% print

#2.3 trControl configurations
trControl = trainControl(method = "repeatedcv",
                         number = 5,
                         classProbs = T,
                         search = "random",
                         allowParallel = T)

#2.4 preProc configurations
preProc = c("scale", "center")

#2.5 Metric configuration
metric = "Accuracy"

#3. 8 Training Models on trainset ----

cl = makePSOCKcluster(5)
registerDoParallel(cl)

#knn
set.seed(seed)
fit.knn = train(formula,
                data = trainset,
                method = "knn",
                preProc = preProc, metric = metric)

#logistic regression
set.seed(seed)
fit.lg = train(formula,
               data = trainset,
               method = "glm",
               family = "binomial",
               preProc = preProc, metric = metric)

#gbm
set.seed(seed)
fit.gbm = train(formula,
                data = trainset,
                method = "gbm",
                preProc = preProc, metric = metric)
#svm
set.seed(seed)
fit.svm = train(formula,
                data = trainset,
                method = "svmRadial",
                preProc = preProc, metric = metric)
#nnet
set.seed(seed)
fit.nnet = train(formula,
                 data = trainset,
                 method = "nnet",
                 preProc = preProc, metric = metric)

stopCluster(cl)

#4. comparing training performance of all 8 models ----
results = resamples(list(knn = fit.knn, lg = fit.lg, 
                         gbm = fit.gbm, svmRad = fit.svm, nnet = fit.nnet))
summary(results)
dot.plot=dotplot(results) %T>% print()
bwplot = bwplot(results) %T>% print()
# I will tune gbm and svmRad, as they have the highest accuracy and Kappa stats.

#5. Tuning 2 models ----

#5.2 gbm
#Tune grid
print(fit.gbm) 
#used n.trees = 50, interaction.depth =2, shrinkage = 0.1 and n.minobsinnode = 10
getModelInfo("gbm")
tunegrid.gbm = expand.grid(
  .n.trees = c(50,100),
  .interaction.depth = 2,
  .shrinkage = 0.1,
  .n.minobsinnode= c(5,10,15)
)
#Training tuned model

cl = makePSOCKcluster(5)
registerDoParallel(cl)

set.seed(seed)
tune.gbm = train(formula,
                 data = trainset,
                 method = "gbm",
                 metric = metric, preProc = preProc,
                 trControl = trControl, tuneGrid = tunegrid.gbm)

stopCluster(cl)
#Checking train performance
print(tune.gbm) #n.trees = 50, interaction.depth = 2, shrinkage = 0.1 and n.minobsinnode = 5
tune.gbm %>% getTrainPerf()
#confusion matrix
tune.gbm %>% attributes()
tune.gbm %>% confusionMatrix.train() # Accuracy (average) : 0.9424

#5.3 svm
#Tune grid
print(fit.svm) #used sigma = sigma = 0.006111922 and C = 1
getModelInfo("svmRadial")
tunegrid.svm = expand.grid(
  .sigma = c(0.05 : 0.07),
  .C = c(0.5, 1, 1.5)
)
#Training tuned model

cl = makePSOCKcluster(5)
registerDoParallel(cl)

set.seed(seed)
tune.svm = train(formula,
                 data = trainset,
                 method = "svmRadial",
                 metric = metric,  preProc = preProc,
                 trControl = trControl, tuneGrid = tunegrid.svm)
stopCluster(cl)
#Checking train performance
print(tune.svm) #sigma = 0.05 and C = 0.5
tune.svm %>% getTrainPerf()
#confusion matrix
tune.svm %>% confusionMatrix.train() # Accuracy (average) : 0.9172

# 6. Tuned model performance on training set ----

#6.1 confusion matrix (%)
tune.gbm %>% confusionMatrix.train() %>% .$table
tune.svm %>% confusionMatrix.train() %>% .$table

#6.2 train performance
tune.gbm %>% getTrainPerf()
tune.svm %>% getTrainPerf()

#7. Tuned model performance on testing set ----

#7.1 confusion matrix (counts)
cm_counts = function(model) {
  a= predict(model, testset)
  b= confusionMatrix(a, testset[[target.label]])
  print(b)
}

test.con.gbm = cm_counts(tune.gbm)
test.con.svm = cm_counts(tune.svm)

test.con.gbm %>% .$table
test.con.svm %>% .$table


#7.2 Confusion Matrix (%)
cm_perc = function(cm) {
  (prop.table(cm$table))*100
}

cm_perc(test.con.gbm)
cm_perc(test.con.svm)

#8. Comparing trainset performance & testset performance ----
#8.1 Trainset performance
trainperf = function(model) {
  a=select(getTrainPerf(model), -c(TrainKappa,method))
  print(a)
}

train.gbm = trainperf(tune.gbm) 
train.svm = trainperf(tune.svm) 

#8.2 Testset performance
testperf  = function(cm) {
  print(cm$overall[c("Accuracy")])
}

test.gbm = testperf(test.con.gbm)
test.svm = testperf(test.con.svm)

result.table = bind_rows(
  c(train.gbm, test.gbm),
  c(train.svm, test.svm)
) %>% data.frame() %>% 
  set_rownames(c("gbm", "svm")) %>% 
  set_colnames(c("accuracy.train", "accuracy.test"))

#9. Final Analysis ----
print(result.table)

#9.1 Analysis on finding the best model
#In this analysis, we are only going to take accuracy into account for determining the best model.
#From intial models prior to tuning, gbm and svmRad performed the best. However, after tuning the
#two models, gbm performed better than svm both on training and testing sets.

#9.2 Analysis on the data
test.con.gbm %>% .$table
cm_perc(test.con.gbm)
#What percentage of introverts from all introverts are correctly predicted?
#(73/76)*100 =  96.05263 (%)
#What percentage of extroverts from all extroverts are incorrectly predicted?
# (2/15)*100 = 13.33333 (%)