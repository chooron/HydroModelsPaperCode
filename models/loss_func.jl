using Statistics

mse(y, y_hat) = mean((y .- y_hat) .^ 2)

rse(y, y_hat) = sum((y .- y_hat) .^ 2) / sum((y .- mean(y)) .^ 2)

nse(y, y_hat) = 1 - sum((y .- y_hat) .^ 2) / sum((y .- mean(y)) .^ 2)


export mse, rse, nse
