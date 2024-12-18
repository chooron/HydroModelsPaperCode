using JLD2
using Plots
using Statistics

# Load the optimization results
opt_loss_df = load("src/implements/save/exphydro_opt.jld2", "loss_df")
default(fontfamily="calibri")
# Calculate moving average
function moving_average(data, window_size)
    output = similar(data)
    for i in 1:length(data)
        start_idx = max(1, i - window_size + 1)
        output[i] = mean(data[start_idx:i])
    end
    return output
end

# Create plot with customizations
p = plot(opt_loss_df.loss,
    label="Original Loss",
    color=:gray,
    alpha=0.4,
    xlabel="Iteration",
    ylabel="MSE",
    legendfont=font("calibri", 10),
    tickfont=font("calibri", 10),
    guidefont=font("calibri", 12),
    xticks=0:2000:length(opt_loss_df.loss),
    dpi=300)

# Add moving average
window_size = 100  # Adjust this value to change the smoothing level
ma_loss = moving_average(opt_loss_df.loss, window_size)
plot!(ma_loss,
    label="Moving Average (n=$window_size)",
    color=:red,
    linewidth=2,
    alpha=0.8)

# Save the figure
savefig(p, "src/implements/plot/figures/exphydro_training.png")