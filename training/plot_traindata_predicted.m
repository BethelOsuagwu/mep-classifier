load traindata_predicted.mat
plot(data(1:end))
hold
plot(predict(1:end,:))