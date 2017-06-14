rm(list=ls(all=TRUE))

library(h2o)
h2o.init(nthreads=-1, max_mem_size="2G")
h2o.removeAll() ## clean slate - just in case the cluster was already running
h2o.no_progress()  # Disable progress bars for Rmd

train_file <- "mnist_train.csv.gz"
test_file <- "mnist_test.csv.gz"

train <- h2o.importFile(train_file)
test <- h2o.importFile(test_file)

y <- paste("C",ncol(train),sep = "")  #the column name of label
x <- setdiff(names(train), y)  #take the data without its label

# Since the response is encoded as integers, we need to tell H2O that
# the response is in fact a categorical/factor column.  Otherwise, it 
# will train a regression model instead of multiclass classification.
train[,y] <- as.factor(train[,y])
test[,y] <- as.factor(test[,y])

dl_fit1 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "dl_fit1",
                            hidden = c(20,20),
                            seed = 1)

dl_fit2 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "dl_fit2",
                            epochs = 50,
                            hidden = c(20,20),
                            stopping_rounds = 0,  # disable early stopping
                            seed = 1)

dl_fit3 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "dl_fit3",
                            epochs = 50,
                            hidden = c(20,20),
                            nfolds = 3,                            #used for early stopping
                            score_interval = 1,                    #used for early stopping
                            stopping_rounds = 5,                   #used for early stopping
                            stopping_metric = "misclassification", #used for early stopping
                            stopping_tolerance = 1e-3,             #used for early stopping
                            seed = 1)


dl_perf1 <- h2o.performance(model = dl_fit1, newdata = test)
dl_perf2 <- h2o.performance(model = dl_fit2, newdata = test)
dl_perf3 <- h2o.performance(model = dl_fit3, newdata = test)

# Retreive test set MSE
h2o.mse(dl_perf1)
h2o.mse(dl_perf2) 
h2o.mse(dl_perf3) 

h2o.scoreHistory(dl_fit3)
plot(dl_fit3, 
     timestep = "epochs", 
     metric = "classification_error")


##Autoencoders in H2O

splits <- h2o.splitFrame(train, 0.5, seed = 1)

# first part of the data, without labels for unsupervised learning
train_unsupervised <- splits[[1]]

# second part of the data, with labels for supervised learning
train_supervised <- splits[[2]]

dim(train_supervised)
dim(train_unsupervised)
hidden <- c(128, 64, 128)
ae_model <- h2o.deeplearning(x = x, 
                             training_frame = train_unsupervised,
                             model_id = "mnist_autoencoder",
                             ignore_const_cols = FALSE,
                             activation = "Tanh",  # Tanh is good for autoencoding
                             hidden = hidden,
                             autoencoder = TRUE)

fit2 <- h2o.deeplearning(x = x, y = y,
                         training_frame = train_supervised,
                         ignore_const_cols = FALSE,
                         hidden = hidden)
perf2 <- h2o.performance(fit2, newdata = test)
h2o.mse(perf2)


##ANOMALI DETECTION

test_rec_error <- as.data.frame(h2o.anomaly(ae_model, test)) 
test_recon <- predict(ae_model, test)

# helper functions for display of handwritten digits
plotDigit <- function(mydata, rec_error) {
  len <- nrow(mydata)
  N <- ceiling(sqrt(len))
  par(mfrow = c(N,N), pty = 's', mar = c(1,1,1,1), xaxt = 'n', yaxt = 'n')
  for (i in 1:nrow(mydata)) {
    colors <- c('white','black')
    cus_col <- colorRampPalette(colors = colors)
    z <- array(mydata[i,], dim = c(28,28))
    z <- z[,28:1]
    class(z) <- "numeric"
    image(1:28, 1:28, z, main = paste0("rec_error: ", round(rec_error[i],4)), col = cus_col(256))
  }
}
plotDigits <- function(data, rec_error, rows) {
  row_idx <- sort(order(rec_error[,1],decreasing=F)[rows])
  my_rec_error <- rec_error[row_idx,]
  my_data <- as.matrix(as.data.frame(data[row_idx,]))
  plotDigit(my_data, my_rec_error)
}

plotDigits(test_recon, test_rec_error, c(1:6))
plotDigits(test, test_rec_error, c(1:6))

#references
#https://htmlpreview.github.io/?https://github.com/ledell/sldm4-h2o/blob/master/sldm4-deeplearning-h2o.html
# adapted from http://www.r-bloggers.com/the-essence-of-a-handwritten-digit/