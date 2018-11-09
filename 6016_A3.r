# 6016 Assignment 3
# Author: LAM Hin Tai
# UID: 2004062587
#
library(ggplot2)
library(spatstat)
library(sp)

h <- readRDS('P:/__personal/MStat/_S4 6016/A3/houses2000.rds')

#### Parts (a) to (g) - visualization ####
# (a) - Store coordinates from every polygon in a dataframe
polygonDf <- fortify(h)
str(polygonDf)

# (b) - Transform data
houseDf <- h@data
houseDf$logHouseValue <- log(houseDf$houseValue + 1)

houseDf$yearsSinceBuilt <- 2000 - houseDf$yearBuilt

# There are missing data - some houses do not have the yearBuilt
# Mark them as NA
sum(houseDf$yearBuilt==0)
houseDf$yearsSinceBuilt[houseDf$yearBuilt==0] <- NA

# Transform each count variable to sqrt(count)
houseDf$

#### Parts (h) to (n) - Prediction model for houseValue ####

