# 6016 Assignment 2
# Author: LAM Hin Tai
# UID: 2004062587
#
library(ggplot2)
library(gstat)
library(geostatsp)
data(swissRain)

# Q1 - Provided code from the question
# This code sets up a color ramp scale from red to blue and returns a function
# rbPal.
# The rbPal function in turn takes an interger argument, 10, and returns a
# character vector of length 10.
# The character vector from rbPal stores RGB character codes of 10 colors that
# are created by interpolating between the 2 poles, red and blue.
# The rainfall data from swissRain@data$rain is then taken and its whole range
# divided into 10 intervals. Each data point in swissRain@data$rain is then
# converted into an index number corresponding to which interval the data point
# is in. These index number then gets assigned to the corresponding RGB code
# and stored into Col.
# Afterwards, the swissAltitude data is plotted as a graph, with title "Swiss
# Rainfall with background elevation", and bullet points added onto the graph
# according to swissRain@coords and the corresponding colors according to Col.
# Locations with the lowest rainfall are plotted as red bullet points while
# those with the highest rainfall as blue bullet points.
# Finally, the swiss border is also plotted onto the same graph from swissBorder.
rbPal <- colorRampPalette(c('red','blue'))
Col <- rbPal(10)[as.numeric(cut(swissRain@data$rain,breaks = 10))]
plot(swissAltitude, main="Swiss Rainfall with background elevation")
points(swissRain,pch = 20,col = Col)
plot(swissBorder, add=TRUE)

# Q2 - Log transform the rainfall data
swissRain@data$logRain <- log(swissRain@data$rain)

#### Variogram ####
# Q3 - Empirical variogram of the rainfall
# Distance matrix S
distMatS <- ecodist::full(ecodist::distance(swissRain@coords, "euclidean"))
rainGsObj <- gstat(g=NULL, id="res", formula=logRain~1, data=swissRain)
# Empirical variogram
vempRain <- variogram(object=rainGsObj, cutoff=max(distMatS)/2, width=13000)
vempRain

# Q4 - Fit variogram model to the empirical variogram
vfitRain <- fit.variogram(object=vempRain, model=vgm(c("Exp", "Sph", "Mat", "Gau")))
vfitRain

# These will fit without nugget effect
vfitRain_Exp <- fit.variogram(object=vempRain, model=vgm(model="Exp"))
vfitRain_Exp

vfitRain_Sph <- fit.variogram(object=vempRain, model=vgm(model="Sph"))
vfitRain_Sph

vfitRain_Mat <- fit.variogram(object=vempRain, model=vgm(model="Mat"))
vfitRain_Mat

vfitRain_Gau <- fit.variogram(object=vempRain, model=vgm(model="Gau"))
vfitRain_Gau

# Specifically include nugget effect
vfitRain_ExpNug <- fit.variogram(object=vempRain, model=vgm("Exp"))
vfitRain_ExpNug

vfitRain_SphNug <- fit.variogram(object=vempRain, model=vgm("Sph"))
vfitRain_SphNug

vfitRain_MatNug <- fit.variogram(object=vempRain, model=vgm("Mat"))
vfitRain_MatNug

vfitRain_GauNug <- fit.variogram(object=vempRain, model=vgm("Gau"))
vfitRain_GauNug

# Determine quality by least squares
attributes(vfitRain)$SSErr
attributes(vfitRain_Exp)$SSErr
attributes(vfitRain_Sph)$SSErr
attributes(vfitRain_Mat)$SSErr
attributes(vfitRain_Gau)$SSErr
attributes(vfitRain_ExpNug)$SSErr
attributes(vfitRain_SphNug)$SSErr
attributes(vfitRain_MatNug)$SSErr
attributes(vfitRain_GauNug)$SSErr

# Q5 - Plot the empirical variogram and fitted variogram
plot(vempRain, vfitRain, main="Empirical variogram and the fitted model")

#### Kriging for the mean ####
# Q6 - Estimate the mean and its variance of the rainfall
sphericalVar <- function(sill, nugget, range, h) {
  if (h == 0) {
    return(0)
  } else if (h>range) {
    return(sill)
  } else {
    return(nugget + (sill-nugget) * (3*h/(2*range) - 0.5*h^3/range^3))
  }
}
calSphVariogram <- function(j, i, vfit, S) {
  return(sphericalVar(vfit$psill[2], vfit$psill[1], vfit$range[2], S[i,j]))
}

sphVariogramMat <- matrix(0, nrow=nrow(distMatS), ncol=ncol(distMatS))
for (i in 1:nrow(distMatS)) {
  sphVariogramMat[i,] <- sapply(1:nrow(distMatS), calSphVariogram, i, vfitRain, distMatS)
}
sphCov <- vfitRain$psill[2] - sphVariogramMat
ones <- rep(1,nrow(distMatS))
sphCovInv <- qr.solve(sphCov)

# Weight vector for kriging for the mean
w <- sphCovInv %*% ones / ((ones %*% sphCovInv %*% ones)[1,1])
meanLogRain <- sum(w * swissRain$logRain)
meanLogRain

# Q7 - Divide the data into a training set and a testing set
n <- length(swissRain)
trainSetRatio = 0.75
set.seed(6779) # make the sample reproducible
trainCaseIds <- sample(1:n, trainSetRatio*n, replace=FALSE)

trainDataSet <- swissRain[trainCaseIds,]
testDataSet <- swissRain[-trainCaseIds,]

str(trainDataSet)
str(testDataSet)

testDataSet$ID
testDataSet$logRain
testDataSet@coords

#### Simple Kriging ####
# Q8 - Predict the rainfall on the testing data set
# rainMeanGsObj <- gstat(g=rainGsObj, id="res", formula=logRain~1, data=swissRain)
n_train <- length(trainDataSet)
distMatS_train <- ecodist::full(ecodist::distance(trainDataSet@coords, "euclidean"))
rainGsObj_train <- gstat(g=rainGsObj, id="resSK", formula=logRain~1, data=trainDataSet, model=vfitRain, beta=mean(swissRain$logRain))



