using CSV, DataFrames, Dates, ComponentArrays
using Optimization, OptimizationOptimisers
using DataInterpolations, SciMLSensitivity
using Statistics, StableRNGs
using HydroModels

include("../../models/m50v2.jl")

df = DataFrame(CSV.File("data/m50/01013500.csv"))
ts = collect(1:10000)
prcp_vec = df[ts, "Prcp"]
temp_vec = df[ts, "Temp"]
dayl_vec = df[ts, "Lday"]
snowpack_vec = df[ts, "SnowWater"]
soilwater_vec = df[ts, "SoilWater"]
flow_vec = df[ts, "Flow"]
qobs_vec = DataFrame(CSV.File("data/exphydro/01013500.csv"))[ts, "flow(mm)"]

inputs = [prcp_vec, temp_vec, snowpack_vec, soilwater_vec]
means, stds = mean.(inputs), std.(inputs)
(prcp_norm_vec, temp_norm_vec, snowpack_norm_vec, soilwater_norm_vec) = [
    @.((inp - mean) / std) for (inp, mean, std) in zip(inputs, means, stds)
]
norm_input = (
    norm_prcp=prcp_norm_vec, norm_temp=temp_norm_vec,
    norm_snw=snowpack_norm_vec, norm_slw=soilwater_norm_vec
)

mse(y, y_hat) = mean((y .- y_hat) .^ 2)
rse(y, y_hat) = sum((y .- y_hat) .^ 2) / sum((y .- mean(y)) .^ 2)
nse(y, y_hat) = 1 - sum((y .- y_hat) .^ 2) / sum((y .- mean(y)) .^ 2)

#* pretrain the evap neural network
#* 1 prepare the input and output and parameters
ep_input_matrix = stack(norm_input[HydroModels.get_input_names(ep_nn_flux)], dims=1)
log_evap_div_lday_vec = log.(df.Evap ./ df.Lday)
epnn_ps = LuxCore.initialparameters(StableRNG(42), ep_nn) |> ComponentVector |> Vector
epnn_ps_cv = ComponentVector(nns=(epnn=epnn_ps,))
epnn_axes = getaxes(epnn_ps_cv)
#* 2 define the loss function
epnn_objective(p, _) = mse(log_evap_div_lday_vec, ep_nn_flux(ep_input_matrix, ComponentVector(p, epnn_axes))[1, :])
#* 3 define the optimization problem
ep_opt_prob = OptimizationProblem(OptimizationFunction(epnn_objective, AutoZygote()), epnn_ps_cv |> Vector)
#* 4 solve the optimization problem
ep_sol = solve(ep_opt_prob, Adam(1e-2), maxiters=1000)

#* pretrain the flow neural network
#* 1 prepare the input and output and parameters
q_input_matrix = stack(norm_input[HydroModels.get_input_names(q_nn_flux)], dims=1)
log_flow_vec = log.(df.Flow)
qnn_ps = LuxCore.initialparameters(StableRNG(42), q_nn) |> ComponentVector |> Vector
qnn_ps_cv = ComponentVector(nns=(qnn=qnn_ps,))
qnn_axes = getaxes(qnn_ps_cv)
#* 2 define the loss function
qnn_objective(p, _) = mse(log_flow_vec, q_nn_flux(q_input_matrix, ComponentVector(p, qnn_axes))[1, :])
#* 3 define the optimization problem
q_opt_prob = OptimizationProblem(OptimizationFunction(qnn_objective, AutoZygote()), qnn_ps_cv |> Vector)
#* 4 solve the optimization problem
q_sol = solve(q_opt_prob, Adam(1e-2), maxiters=1000)

#* train m50 model
#* 1 prepare the input
norm_vars = [:prcp, :temp, :snowpack, :soilwater]
var_stds = NamedTuple{Tuple([Symbol(nm, :_std) for nm in norm_vars])}(stds)
var_means = NamedTuple{Tuple([Symbol(nm, :_mean) for nm in norm_vars])}(means)
params = reduce(merge, [(Df=2.674, Tmax=0.17, Tmin=-2.09), var_means, var_stds])
initstates = ComponentVector(snowpack=0.0, soilwater=1303.00)
pretrained_pas = ComponentVector(params=params, nns=(epnn=ep_sol.u, qnn=q_sol.u))
pas_axes = getaxes(pretrained_pas)
input_mat = stack([temp_vec, dayl_vec, prcp_vec], dims=1)
config = (solver=HydroModels.ODESolver(sensealg=GaussAdjoint(autojacvec=EnzymeVJP())), interp=LinearInterpolation)
#* 3 define the optimization problem
m50_objective(p, _) = rse(qobs_vec, m50_model(input_mat, ComponentVector(p, pas_axes), initstates=initstates, config=config)[end, :])
m50_opt_prob = OptimizationProblem(OptimizationFunction(m50_objective, AutoZygote()), pretrained_pas |> Vector)
#* 4 define the callback function
loss_history = []
callback_func!(state, l) = begin
    push!(loss_history, (iter=state.iter, loss=l, time=now()))
    println("iter: $(state.iter), loss: $l, time: $(now())")
    false
end
#* 5 solve the optimization problem
m50_sol = solve(m50_opt_prob, Adam(1e-2), maxiters=100, callback=callback_func!)

#* plot the results
m50_output = m50_model(input_mat, ComponentVector(m50_sol.u, pas_axes), initstates=initstates, config=config)[end, :]
pretrained_output = exp.(q_nn_flux(q_input_matrix, ComponentVector(q_sol.u, qnn_axes))[1, :])
m50_pretrained_output = m50_model(input_mat, pretrained_pas, initstates=initstates, config=config)[end, :]

plot(ts, m50_output, label="m50", lw=1)
plot!(ts, qobs_vec, label="obs", lw=1)

plot(ts, log.(pretrained_output), label="m50-pretrained", lw=1)
plot!(ts, log_flow_vec, label="log flow", lw=1)

xlabel!("time (day)")
ylabel!("flow (mm/day)")