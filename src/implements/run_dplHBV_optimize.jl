using CSV, DataFrames, JLD2, Dates
using ComponentArrays
using BenchmarkTools
using StableRNGs
using DataInterpolations
using Optimization, OptimizationOptimisers
using SciMLSensitivity
using Plots
using HydroModels
using Zygote

include("../../models/dplHBV.jl")
include("loss_functions.jl")

#* load data
df = DataFrame(CSV.File("data/exphydro/01013500.csv"));
ts = collect(1:10000)
prcp_vec = df[ts, "prcp(mm/day)"]
temp_vec = df[ts, "tmean(C)"]
dayl_vec = df[ts, "dayl(day)"]
qobs_vec = df[ts, "flow(mm)"]
pet_vec = @. 29.8 * dayl_vec * 24 * 0.611 * exp((17.3 * temp_vec) / (temp_vec + 237.3)) / (temp_vec + 273.2)
input = (prcp=prcp_vec, pet=pet_vec, temp=temp_vec)

#* prepare parameters
psnn_ps, psnn_st = Lux.setup(StableRNG(123), params_nn)
psnn_ps_vec = ComponentVector(psnn_ps) |> Vector
params = ComponentVector(TT=0.0, CFMAX=5.0, CWH=0.1, CFR=0.05, FC=200.0, LP=0.6, k0=0.06, k1=0.2, k2=0.1, PPERC=2, UZL=10)
nns = ComponentVector(NamedTuple{Tuple(HydroModels.get_nn_names(dpl_hbv_model))}([psnn_ps_vec]))
init_states = ComponentVector(suz=0.0, slz=0.0, soilwater=0.0, meltwater=0.0, snowpack=0.0)
pas = ComponentVector(params=params, nns=nns)
ps_axes = getaxes(pas)
input_arr = stack(input[HydroModels.get_input_names(dpl_hbv_model)], dims=1)

#* define config
config = (solver=ODESolver(), timeidx=ts, interp=LinearInterpolation)
predict_before_opt = dpl_hbv_model(input_arr, pas, initstates=init_states, config=config)

#* define optimization problem
loss_fn(y, y_hat) = mse(y[365:end], y_hat[end, 365:end])
objective(p, _) = loss_fn(qobs_vec, dpl_hbv_model(input_arr, ComponentVector(p, ps_axes), initstates=init_states, config=config))
optfunc = Optimization.OptimizationFunction(objective, Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optfunc, pas |> Vector)

#* define callback function
records = []
callbackfunc!(state, l) = begin
    push!(records, (iter=state.iter, loss=l, time=now()))
    println("iter: $(state.iter), loss: $l, time: $(now())")
    return false
end
sol = solve(optprob, Adam(), maxiters=200, callback=callbackfunc!)
save(
    "src/implements/save/dplHBV_opt.jld2", # save path
    "opt_params", ComponentVector(sol.u, ps_axes), # optimial parameters
)
CSV.write("src/implements/save/dplHBV_opt_records.csv", DataFrame(records))
pred_flow = dpl_hbv_model(input_arr, ComponentVector(sol.u, ps_axes), initstates=init_states, config=config)[end, :]

plot(ts, pred_flow, label="pred", lw=1)
plot!(ts, qobs_vec, label="obs", lw=1)
xlabel!("time (day)")
ylabel!("flow (mm/day)")
title!("dplHBV optimize")
savefig("src/implements/plot/figures/dplHBV_opt.png")