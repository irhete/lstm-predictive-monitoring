setwd("/home/irene/Repos/lstm-predictive-monitoring/loss_files2/")

library(ggplot2)
library(reshape2)

result_files <- list.files()
result_files


data <- read.table(result_files[1], sep=";", header=T)
for (filename in result_files[-c(1)]) {
  data <- rbind(data, read.table(filename, sep=";", header=T))
}

data <- subset(data, !(grepl("softsign", params)) & !(grepl("1class", params)))

data_m <- melt(data, id.vars=c("epoch", "params", "dataset"))

base_size = 28
line_size = 1
point_size = 3

ggplot(data, aes(x=epoch, y=val_loss-train_loss, color=params)) + geom_point(size=point_size) + geom_line(size=line_size) + theme_bw(base_size=base_size) + facet_wrap(~ dataset, ncol=2)

ggplot(data, aes(x=epoch, y=train_loss, color=params)) + geom_point(size=point_size) + geom_line(size=line_size) + theme_bw(base_size=base_size) + facet_wrap(~ dataset, ncol=2)

ggplot(data, aes(x=epoch, y=val_loss, color=params)) + geom_point(size=point_size) + geom_line(size=line_size) + theme_bw(base_size=base_size) + facet_wrap(~ dataset, ncol=2)
ggplot(subset(data, grepl("complete", params)), aes(x=epoch, y=val_loss, color=params)) + geom_point(size=point_size) + geom_line(size=line_size) + theme_bw(base_size=base_size) + facet_wrap(~ dataset, ncol=2)



ggplot(subset(data_m, grepl("complete", params)), aes(x=epoch, y=value, color=variable)) + geom_point(size=point_size) + geom_line(size=line_size) + theme_bw(base_size=base_size) + facet_wrap(~ dataset, ncol=2)


library(plyr)

ddply(data, .(params), summarize, min(train_loss))
ddply(data, .(params), summarize, min(val_loss))
