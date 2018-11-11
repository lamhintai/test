# 6016 Assignment 3
# Author: LAM Hin Tai
# UID: 2004062587
#
library(ggplot2)
library(spatstat)
library(sp)
library(rdist)
library(iterators)
library(foreach)
library(parallel)
library(doParallel)
library(FNN)


h <- readRDS('C:/Users/Tai/Documents/MStat 2018/S4 6016 Spatial Data Analysis/Assignment 3/houses2000.rds')

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

# # There are missing data - some houses do not have the yearBuilt
# # Mark them as NA
# sum(houseDf$yearBuilt==0)
# houseDf$houseYears[houseDf$yearBuilt==0] <- NA

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
houseDf$sFamilies <- sqrt(houseDf$Families)

# (c) - Backward elimination of regression of log(houseValue+1) on transformed variables
fullModel <- lm(logHouseValue~houseYears+sHousingUn+sRecHouses+sMobileHom+sBadPlumbi+
                  sBadKitche+sPopulation+sMales+sFemales+sUnder5+sWhite+sBlack+
                  sAmericanIn+sAsian+sHispanic+sPopInHouse+sHousehold+sFamilies+
                  nRooms+nBedrooms+medHHinc+MedianAge+householdS+familySize,
                data=houseDf)
lmFit <- step(fullModel, direction="backward")
summary(lmFit)

# (d) - Store the regression residual as u
u <- lmFit$residuals

# (e) - Plot regions coloured by houseValue
centroids <- data.frame(coordinates(h))
colnames(centroids) <- c("X", "Y")

# Calculate distances of every location (coord) and centroid pairs
# to assign location to the region (centroid)
nCores <- detectCores() - 1
cl <- makeCluster(nCores)
registerDoParallel(cl)

regions <- foreach(coord = iter(as.matrix(mapDf[, 1:2]), by='row'), .combine=c,
                   .packages='rdist') %dopar%
  {
    distToCentroid <- cdist(coord, centroids, metric="euclidean")
    region <- which.min(distToCentroid)
  }

stopImplicitCluster()
str(regions)
range(regions)

saveRDS(regions, file="C:/Users/Tai/Documents/MStat 2018/6016 A3/regions.RDS")
regions <- readRDS(file="C:/Users/Tai/Documents/MStat 2018/6016 A3/regions.RDS")

mapDf$houseValue <- houseDf$houseValue[regions]
plotHouseVal <- ggplot(mapDf)+geom_polygon(aes(x=long, y=lat, group=group, fill=houseValue, color=houseValue))+
  geom_path(aes(x=long, y=lat, group=group))+scale_fill_gradient2()+scale_colour_gradient2()+
  #geom_point(data=centroids, aes(x=X, y=Y), size=1, color='blue')+
  coord_fixed()+ guides(fill=FALSE)
plotHouseVal

# (f) - Plot regions coloured by median household income
mapDf$medHHinc <- houseDf$medHHinc[regions]
plotMedHHInc <- ggplot(mapDf)+geom_polygon(aes(x=long, y=lat, group=group, fill=medHHinc, color=medHHinc))+
  geom_path(aes(x=long, y=lat, group=group))+scale_fill_gradient2()+scale_colour_gradient2()+
  #geom_point(data=centroids, aes(x=X, y=Y), size=1, color='blue')+
  coord_fixed()+ guides(fill=FALSE)
plotMedHHInc

# (g) - Plot regions coloured by residuals of regression
mapDf$residual <- u[regions]
plotResidual <- ggplot(mapDf)+geom_polygon(aes(x=long, y=lat, group=group, fill=residual, color=residual))+
  geom_path(aes(x=long, y=lat, group=group))+scale_fill_gradient2()+scale_colour_gradient2()+
  #geom_point(data=centroids, aes(x=X, y=Y), size=1, color='blue')+
  coord_fixed()+ guides(fill=FALSE)
plotResidual

#### Parts (h) to (n) - Prediction model for houseValue ####
# (h) Row-normalized spatial weight matrices
# 5-nn spatial weight matrix
n <- nrow(centroids)
nn <- get.knn(centroids, k=5)
Wnn <- matrix(0, nrow=n, ncol=n)
for (i in seq_len(n)) {
  Wnn[i, nn$nn.index[i,]] <- 1
  Wnn[nn$nn.index[i,], i] <- 1
}
str(Wnn)
# Row-normalized 5-nn spatial weight matrix
for (i in seq_len(n)) {
  Wnn[i,] <- Wnn[i,]/sum(Wnn[i,])
}
