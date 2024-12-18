using CSV
using Lux
using Random
using DataFrames
using ComponentArrays
using BenchmarkTools
using StableRNGs
using DataInterpolations
using Optimization
using OptimizationOptimisers
using SciMLSensitivity
using JLD2
using Plots
using HydroModels

include("../models/dplHBV.jl")
include("loss_functions.jl")

#* load data
df = DataFrame(CSV.File("data/exphydro/01013500.csv"));
ts = collect(1:10000)
prcp_vec = df[ts, "prcp(mm/day)"]
temp_vec = df[ts, "tmean(C)"]
dayl_vec = df[ts, "dayl(day)"]
pet_vec = @. 29.8 * dayl_vec * 24 * 0.611 * exp((17.3 * temp_vec) / (temp_vec + 237.3)) / (temp_vec + 273.2)
qobs_vec = df[ts, "flow(mm)"]
input = (prcp=prcp_vec, pet=pet_vec, temp=temp_vec)
#* prepare parameters
psnn_ps, psnn_st = Lux.setup(StableRNG(123), params_nn)
psnn_ps_vec = Vector(ComponentVector(psnn_ps))
params = ComponentVector(TT=0.0, CFMAX=5.0, CWH=0.1, CFR=0.05, FC=200.0, LP=0.6, k0=0.06, k1=0.2, k2=0.1, PPERC=2, UZL=10)
nns = ComponentVector(NamedTuple{Tuple([nn_wrapper.meta.name])}([psnn_ps_vec]))
init_states = ComponentVector(suz=0.0, slz=0.0, soilwater=0.0, meltwater=0.0, snowpack=0.0)
pas = ComponentVector(params=params, initstates=init_states, nn=nns)
#* define config
config = (solver=ODESolver(), timeidx=ts, interp=LinearInterpolation)
adapter = NamedTupleIOAdapter(dpl_hbv_model)
predict_before_opt = adapter(input, pas, config=config)

#! prepare flow
tunable_pas = ComponentVector(params=params, nn=(pnn_nwrapper=psnn_ps_vec,))
const_pas = ComponentVector(initstates=init_states)

model_grad_opt = GradOptimizer(component=adapter, maxiters=50, adtype=AutoZygote(), solve_alg=Adam(1e-2))
config = (solver=ODESolver(sensealg=BacksolveAdjoint(autodiff=true)), timeidx=ts, interp=LinearInterpolation)
hbv_hydro_opt_params, loss_df = model_grad_opt(
    [input], [(q=qobs_vec,)],
    tunable_pas=tunable_pas,
    const_pas=const_pas,
    return_loss_df=true,
    config=[config]
)

output = adapter(input, hbv_hydro_opt_params, config=config)
save("src/implements/save/dplHBV_opt.jld2", "loss_df", loss_df, "opt_params", hbv_hydro_opt_params, "predict-before-opt", predict_before_opt, "predict-after-opt", output)