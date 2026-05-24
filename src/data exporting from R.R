install.packages("C:/Users/HP/Downloads/CASdatasets_1.2-1.tar.gz",
                 repos = NULL,
                 type  = "source")
library(CASdatasets)
data(freclaimset3multi9207)
data(freclaimset3fire9207)
data(freclaimset3dam9207)

write.csv(freclaimset3multi9207, "C:/Users/HP/Downloads/freclaimset3multi9207.csv", row.names=FALSE)
write.csv(freclaimset3fire9207,  "C:/Users/HP/Downloads/freclaimset3fire9207.csv",  row.names=FALSE)
write.csv(freclaimset3dam9207,   "C:/Users/HP/Downloads/freclaimset3dam9207.csv",   row.names=FALSE)
