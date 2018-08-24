df = read.csv('train.csv', stringsAsFactors = FALSE)
y = log(df[, 'SalePrice'])
df = df[, 2:21]

boxplot(y~df[,1]); print(sum(is.na(df[,1]))/length(df[,1])) # MSSubClass
boxplot(y~df[,2]); print(sum(is.na(df[,2]))/length(df[,2])) # MSZoning
plot(y~df[,3]); print(sum(is.na(df[,3]))/length(df[,3]))    # LotFrontage
plot(y~df[,4]); print(sum(is.na(df[,4]))/length(df[,4]))    # LotArea
boxplot(y~df[,5]); print(sum(is.na(df[,5]))/length(df[,5])) # Street
boxplot(y~df[,6]); print(sum(is.na(df[,6]))/length(df[,6])) # Alley
boxplot(y~df[,7]); print(sum(is.na(df[,7]))/length(df[,7])) # LotShape
boxplot(y~df[,8]); print(sum(is.na(df[,8]))/length(df[,8])) # LandContour
boxplot(y~df[,9]); print(sum(is.na(df[,9]))/length(df[,9])) # Utilities
boxplot(y~df[,10]); print(sum(is.na(df[,10]))/length(df[,10]))# LotConfig
boxplot(y~df[,11]); print(sum(is.na(df[,11]))/length(df[,11]))# LandSlope
boxplot(y~df[,12]); print(sum(is.na(df[,12]))/length(df[,12]))# Neighborhood
boxplot(y~df[,13]); print(sum(is.na(df[,13]))/length(df[,13]))# Condition_1
boxplot(y~df[,14]); print(sum(is.na(df[,14]))/length(df[,14]))# Condition_2
boxplot(y~df[,15]); print(sum(is.na(df[,15]))/length(df[,15]))# BldgType
boxplot(y~df[,16]); print(sum(is.na(df[,16]))/length(df[,16]))# HouseStyle
boxplot(y~df[,17]); print(sum(is.na(df[,17]))/length(df[,17]))# OverallQual
boxplot(y~df[,18]); print(sum(is.na(df[,18]))/length(df[,18]))# OverallCond
plot(y~df[,19]); print(sum(is.na(df[,19]))/length(df[,19]))# YearBuilt
plot(y~df[,20]); print(sum(is.na(df[,20]))/length(df[,20]))# YearRemodAdd


# BoxCox stuff
library(forecast)

lambda = BoxCox.lambda(df$ExterQual)
trans.vector = BoxCox(df$ExterQual, lambda)

model = lm(y~df$ExterQual)
plot(y~df$ExterQual)
abline(model)

model = lm(y~trans.vector)
plot(y~trans.vector)
abline(model)