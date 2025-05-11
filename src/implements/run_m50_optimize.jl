using CSV, DataFrames, Dates, ComponentArrays, JLD2
using Optimization, LineSearches
using OptimizationOptimisers: Adam
using OptimizationOptimJL: LBFGS
using DataInterpolations, SciMLSensitivity
using Statistics, StableRNGs
using HydroModels

include("../../models/m50_slight.jl")
include("../../models/loss_func.jl")

data = CSV.read("data/exphydro/01013500.csv", DataFrame)
ts = collect(1:10000)
input = (lday=data[ts, "dayl(day)"], temp=data[ts, "tmean(C)"], prcp=data[ts, "prcp(mm/day)"])
flow_vec = data[ts, "flow(mm)"]
load_exphydro = load("src/implements/save/exphydro_opt.jld2")
exphydro_output = load_exphydro["output"]
exphydro_opt_params = load_exphydro["opt_params"]["params"]
exphydro_initstates = load_exphydro["opt_params"]["initstates"]

inputs = [exphydro_output.prcp, exphydro_output.temp, exphydro_output.snowpack, exphydro_output.soilwater]
means, stds = mean.(inputs), std.(inputs)
norm_input = NamedTuple{(:norm_prcp, :norm_temp, :norm_snw, :norm_slw)}([
    @.((inp - me) / st) for (inp, me, st) in zip(inputs, means, stds)
])

#* pretrain the evap neural network
#* 1 prepare the input and output and parameters
ep_input_matrix = stack(norm_input[HydroModels.get_input_names(ep_nn_flux)], dims=1)
log_evap_div_lday_vec = log.(exphydro_output.evap ./ exphydro_output.lday)
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
log_flow_vec = log.(exphydro_output.flow)
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
params = reduce(merge, [NamedTuple(exphydro_opt_params), var_means, var_stds])
pretrained_pas = ComponentVector(params=params, nns=(epnn=ep_sol.u, qnn=q_sol.u))
pas_axes = getaxes(pretrained_pas)
input_mat = stack(input[HydroModels.get_input_names(m50_model)], dims=1)
config = (solver=HydroModels.ODESolver(sensealg=GaussAdjoint(autojacvec=EnzymeVJP())), interp=LinearInterpolation)
#* 3 define the optimization problem
m50_objective(p, _) = begin
    rse(
        flow_vec,
        m50_model(
            input_mat, ComponentVector(p, pas_axes),
            initstates=exphydro_initstates, config=config
        )[end, :]
    )
end
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
#* 6 more training
more_m50_opt_prob = OptimizationProblem(OptimizationFunction(m50_objective, AutoZygote()), m50_sol.u)
more_m50_sol = solve(more_m50_opt_prob, LBFGS(linesearch = BackTracking()), maxiters=20, callback=callback_func!)

#* plot the results
m50_output = m50_model(input_mat, ComponentVector(more_m50_sol.u, pas_axes), initstates=exphydro_initstates, config=config)[end, :]
pretrained_output = exp.(q_nn_flux(q_input_matrix, ComponentVector(q_sol.u, qnn_axes))[1, :])
m50_pretrained_output = m50_model(input_mat, pretrained_pas, initstates=exphydro_initstates, config=config)[end, :]

@info nse(flow_vec, m50_output)
@info nse(flow_vec, pretrained_output)
@info nse(flow_vec, m50_pretrained_output)

# plot(ts, m50_output, label="m50", lw=1)
# plot!(ts, qobs_vec, label="obs", lw=1)


# xlabel!("time (day)")
# ylabel!("flow (mm/day)")