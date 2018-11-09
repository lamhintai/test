# 6016 Assignment 3
# Author: LAM Hin Tai
# UID: 2004062587
#
library(ggplot2)
library(spatstat)
library(sp)
library(rdist)

h <- readRDS('P:/__personal/MStat/_S4 6016/A3/houses2000.rds')

#### Parts (a) to (g) - visualization ####
# (a) - Store coordinates from every polygon in a dataframe
mapDf <- fortify(h)
str(mapDf)
str(mapDf$long)
str(mapDf$lat)

# (b) - Transform data
houseDf <- h@data

houseDf$logHouseValue <- log(houseDf$houseValue + 1)
houseDf$houseYears <- 2000 - houseDf$yearBuilt

# There are missing data - some houses do not have the yearBuilt
# Mark them as NA
sum(houseDf$yearBuilt==0)
houseDf$houseYears[houseDf$yearBuilt==0] <- NA

# Transform each count variable to sqrt(count)
houseDf$sHousingUn <- sqrt(houseDf$nhousingUn)
houseDf$sRecHouses <- sqrt(houseDf$recHouses)
houseDf$sMobileHom <- sqrt(houseDf$nMobileHom)
houseDf$sBadPlumbi <- sqrt(houseDf$nBadPlumbi)
houseDf$sBadKitche <- sqrt(houseDf$nBadKitche)
houseDf$sPopulation <- sqrt(houseDf$Population)
houseDf$sMales <- sqrt(houseDf$Males)
houseDf$sFemales <- sqrt(houseDf$Females)
houseDf$sUnder5 <- sqrt(houseDf$Under5)
houseDf$sWhite <- sqrt(houseDf$White)
houseDf$sBlack <- sqrt(houseDf$Black)
houseDf$sAmericanIn <- sqrt(houseDf$AmericanIn)
houseDf$sAsian <- sqrt(houseDf$Asian)
houseDf$sHispanic <- sqrt(houseDf$Hispanic)
houseDf$sPopInHouse <- sqrt(houseDf$PopInHouse)
houseDf$sHousehold <- sqrt(houseDf$nHousehold)


# (c) - Backward elimination of regression of log(houseValue+1) on transformed variables
fullModel <- lm(logHouseValue~houseYears+sHousingUn+sRecHouses+sMobileHom+sBadPlumbi+
                  sBadPlumbi+sBadKitche+sPopulation+sMales+sFemales+sUnder5+sWhite+
                  sBlack+sAmericanIn+sAsian+sHispanic+sPopInHouse+sHousehold+nRooms+
                  nBedrooms+medHHinc+MedianAge+householdS+familySize, data=houseDf)
lmFit <- step(fullModel, direction="backward")
summary(lmFit)

# (d) - Store the regression residual as u
u <- lmFit$residuals

# (e) - Plot regions coloured by houseValue
centroids <- data.frame(coordinates(h))
colnames(centroids) <- c("X", "Y")

#calculate distances of every location and centroid pairs
DS <- cdist(centroids[,1:2], mapDf[,1:2], metric="euclidean")
str(DS)

regions <- vector("integer", length=nrow(mapDf))
# for (i in seq_len(mapDf))

# (f) - Plot regions coloured by median household income

# (g) - Plot regions coloured by residuals of regression

#### Parts (h) to (n) - Prediction model for houseValue ####

