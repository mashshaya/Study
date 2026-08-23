###DEPENDENCIES###

install.packages("readr")
install.packages("dplyr")
install.packages("tidyverse")
install.packages("naniar")
install.packages("writexl")
install.packages('MASS')
install.packages("mvnmle")
install.packages("ordinal")
install.packages('broom')
install.packages('memisc')
install.packages('pROC')
install.packages('car')

library(readr)
library(dplyr)
library(tidyverse)
library(naniar)
library(writexl)
library(MASS)
library(mvnmle)
library(ordinal)
library(broom)
library(memisc)
library(pROC)
library(car)


###LOADING DATA###

data = read.csv('diabetes.csv')

summary(data)

###PREPROCESSING###

#changing 0 to NA
data[["Glucose"]][data[["Glucose"]] == 0] <- NA
data[["BloodPressure"]][data[["BloodPressure"]] == 0] <- NA
data[["SkinThickness"]][data[["SkinThickness"]] == 0] <- NA
data[["Insulin"]][data[["Insulin"]] == 0] <- NA
data[["BMI"]][data[["BMI"]] == 0] <- NA

#changing NA to mean
data$Glucose[is.na(data$Glucose)] <- mean(data$Glucose, na.rm = TRUE)
data$BloodPressure[is.na(data$BloodPressure)] <- mean(data$BloodPressure, na.rm = TRUE)
data$SkinThickness[is.na(data$SkinThickness)] <- mean(data$SkinThickness, na.rm = TRUE)
data$Insulin[is.na(data$Insulin)] <- mean(data$Insulin, na.rm = TRUE)
data$BMI[is.na(data$BMI)] <- mean(data$BMI, na.rm = TRUE)


###DESCRIPITVE STATISTICS###

summary(data)

summary(data$Pregnancies)
summary(data$Glucose)
summary(data$Age)

###probit model###

model_1 = glm(Outcome ~ . ,data=data, family = binomial(link = 'probit') )
summary(model_1)

broom::glance(model_1)
memisc::mtable(model_1)

wald_test_result = car::linearHypothesis(model_1, 
                                          c("BloodPressure = 0", "Insulin = 0"))
wald_test_result


probitScalar = mean(dnorm(predict(model_1, type = 'link')))
probitScalar * coef(model_1)

predictModel = predict(model_1, type = 'response')
observed = table(data$Outcome, predictModel > 0.5)
chi_sq = chisq.test(observed)
chi_sq
summary(predictModel)

rocInstance = roc(data$Outcome, predictModel)
auc(rocInstance)
plot(rocInstance, main = "ROC Curve")

table(true = data$Outcome, prediction = round(fitted(model_1)))


model_1_0 = update(model_1, formula = Outcome ~ 1)
MCFADDEN = 1 - as.vector(logLik(model_1) / logLik(model_1_0))
MCFADDEN


###logit-model###