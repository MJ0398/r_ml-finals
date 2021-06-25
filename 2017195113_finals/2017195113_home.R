library(tidyverse)
library(magrittr)
library(outliers)
library(caret)
library(doParallel)

seed = 323

#1. Data Preprocessing ----
data.raw = read.csv("homevalues.csv")

#1.1 Data overview
str(data.raw)
data.raw %>% view()
data.raw %>% sapply(., class) #rm tract, lon, lat, chas

#removing dummy variable
data.raw %<>% select(-c(tract, lon, lat, chas))
data.raw %>% names()

#1.2 check NA / Impute NA
data.raw %>% anyNA

#1.3 Data Cleaning (outliers, Typos)
#Descriptive Stats
summary(data.raw)
outlier(data.raw)
data.raw %>% Hmisc::hist.data.frame() #all outliers ok, but should change lon
data.raw %>% str()

#changing loan
data.raw$lon %<>%
  substr(2, 6) %>% as.numeric
data.raw$lon

#final check
data.raw %>% str()

#2. Splitting Data & Configurations  ----

#2.1 Splitting Data
target.label = "homevalue"
set.seed(seed)
train.index = createDataPartition(data.raw[[target.label]], p = 0.9, list = F)
trainset = data.raw[train.index,]
testset = data.raw[-train.index,]
#checking training & testing data formation
trainset[[target.label]] %>% densityplot()
testset[[target.label]] %>% densityplot() #similar enough

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
                         repeats = 3,
                         search = "random",
                         allowParallel = T)

#2.4 preProc configurations
preProc = c("scale", "center")

#2.5 Metric configuration
metric = "RMSE"

#3. 8 Training Models on trainset ----

cl = makePSOCKcluster(5)
registerDoParallel(cl)

#knn
set.seed(seed)
fit.knn = train(formula,
                data = trainset,
                method = "knn",
                preProc = preProc)
#logistic regression
set.seed(seed)
fit.lg = train(formula,
               data = trainset,
               method = "glm",
               family = "binomial",
               preProc = preProc)

#random forest
set.seed(seed)
fit.rf = train(formula,
               data = trainset,
               method = "rf",
               preProc = preProc)
#gbm
set.seed(seed)
fit.gbm = train(formula,
                data = trainset,
                method = "gbm",
                preProc = preProc)
#svm
set.seed(seed)
fit.svm = train(formula,
                data = trainset,
                method = "svmRadial",
                preProc = preProc)
#nnet
set.seed(seed)
fit.nnet = train(formula,
                 data = trainset,
                 method = "nnet",
                 preProc = preProc)

stopCluster(cl)

#4. comparing training performance of all 8 models ----
results = resamples(list(knn = fit.knn,
                         rf = fit.rf, gbm = fit.gbm, svmRad = fit.svm, nnet = fit.nnet))
summary(results)
dot.plot=dotplot(results) %T>% print()
bwplot = bwplot(results) %T>% print()
# I will tune nnet and svmRad (instead of knn, bc svm better performs when tuned than knn)

#5. Tuning 4 models using train function ----

#5.3 svm
#Tune grid
print(fit.svm) #used sigma = 0.05878653 and C = 1
getModelInfo("svmRadial")
tunegrid.svm = expand.grid(
  .sigma = c(0.04 : 0.07),
  .C = c(0.5, 1, 1.5)
)
#Training tuned model
set.seed(seed)

cl = makePSOCKcluster(5)
registerDoParallel(cl)
tune.svm = train(formula,
                 data = trainset,
                 method = "svmRadial",
                 linout = T,
                 metric = metric,  preProc = preProc,
                 trControl = trControl, tuneGrid = tunegrid.svm)
stopCluster(cl)
#Checking train performance
print(tune.svm)
tune.svm %>% getTrainPerf() #RMSE 3.572557


#5.4 nnet
#Tune grid
print(fit.nnet) #size = 1 and decay = 0
getModelInfo("nnet")
tunegrid.nnet = expand.grid(.size = c(0.5, 1, 1.5),
                            .decay = c(0, 0.05, 0.1))
set.seed(seed)
cl = makePSOCKcluster(5)
registerDoParallel(cl)

tune.nnet = train(formula,
                  data = trainset,
                  method= "nnet",
                  metric = metric, preProc = preProc,
                  linout = T,
                  trControl = trControl, tuneGrid = tunegrid.nnet)
stopCluster(cl)
#Checking train performance
print(tune.nnet)
tune.nnet %>% getTrainPerf() #RMSE  4.330002 

#6. Final Analysis ----
predict(tune.svm, testset) 
predict(tune.nnet, testset)
results1 = resamples(list(tunedsvm = tune.svm, tunednnet = tune.nnet))
summary(results1)
dot.plot=dotplot(results1) %T>% print()
bwplot = bwplot(results1) %T>% print()
#the svm model has a lower RMSE than nnet. Thus, svm predicts better.