using CSV, DataFrames, JLD2, Plots, Statistics
using Lux, StableRNGs, ComponentArrays
using DataInterpolations, OrdinaryDiffEq
using HydroModels
using GlobalSensitivity
include("../../models/exphydro.jl")
# load data
file_path = "data/exphydro/01013500.csv"
data = CSV.File(file_path);
df = DataFrame(data);
ts = collect(1:10000)
q_vec = df[ts, "flow(mm)"]
input = (lday=df[ts, "dayl(day)"], temp=df[ts, "tmean(C)"], prcp=df[ts, "prcp(mm/day)"])
input_arr = stack(input[HydroModels.get_input_names(exphydro_model)], dims=1)
pas = ComponentVector(params=(
    f=0.01674478, Smax=1709.461015, Qmax=18.46996175,
    Df=2.674548848, Tmax=0.175739196, Tmin=-2.092959084
))
pas_axes = getaxes(pas)
initstates = ComponentVector(snowpack=0.0, soilwater=1300.0)
config = (solver=HydroModels.ODESolver(), interp=LinearInterpolation)
output = exphydro_model(input_arr, pas, initstates=initstates, config=config)
nse(y, y_hat) = 1 - sum((y - y_hat).^2) / sum((y .- mean(y)).^2)
function temp_func(p)
    output = exphydro_model(input_arr, ComponentVector(p, pas_axes), initstates=initstates, config=config)[end, :]
    return [mean(output), maximum(output), nse(q_vec, output)]
end
bounds = [[0.0, 0.1], [100.0, 2000.0], [10.0, 50.0], [0.0, 5.0], [0.0, 3.0], [-3.0, 0.0]]
reg_sens = gsa(temp_func, RegressionGSA(true), bounds; samples=1000)
partial_corr = reg_sens.partial_correlation
stand_regr = reg_sens.standard_regression