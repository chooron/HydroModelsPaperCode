using CSV
using DataFrames
using ComponentArrays
using BenchmarkTools
using NamedTupleTools
using Optimization
using OptimizationOptimisers
using JLD2
using Plots
using StableRNGs
using HydroModels
using HydroModelTools

include("../models/HBV.jl")


TOTAL_AREA = 15230.3
convert_eff = 24 * 3600 / (TOTAL_AREA * 1e6) * 1e3
# build model based on HBV and rapid module
@variables q flow flow_routed
@parameters subarea
vector_basin_info = load("src/distributed/data/vector_basin_info.jld2")
vector_dgraph = vector_basin_info["dg8"]
subbasin_areas = vector_basin_info["subbasin_area"]
subbasin_rivlen = vector_basin_info["riv_len_df"]
hybas_id = [Symbol("subbasin_$i") for i in vector_basin_info["hybas_id"]]
flow_convert_flux = HydroFlux([q] => [flow], [subarea], exprs=[q* (subarea * 1e6) / (1000 * 24 * 3600)])
rapid_route = RapidRoute([flow] => [flow_routed], network=vector_dgraph)
hbv_rapid = HydroModel(name=:rapid_hbv, components=[hbv_model.components..., flow_convert_flux, rapid_route])
ioadapter = NamedTupleIOAdapter(hbv_rapid)
# load data
select_idx = 1:14610
time_idx = collect(1:length(select_idx))
vector_meto = load("src/distributed/data/vector_meto.jld2")
flow_df = CSV.read("src/distributed/data/flow.csv", DataFrame)
flow_vec = flow_df[select_idx, "FLOW"]
subbasin_pet_df = vector_meto["pet"][select_idx, :]
subbasin_prcp_df = vector_meto["prec"][select_idx, :]
subbasin_temp_df = vector_meto["temp"][select_idx, :]
subbasin_type_ids = CSV.read("src/distributed/data/subbasin_types.csv", DataFrame)[!, :SUBBASIN_TYPE]
subbasin_types = [Symbol("subtype_$i") for i in subbasin_type_ids]
subbasin_areas = load("src/distributed/data/vector_basin_info.jld2")["subbasin_area"]
subtype_category = unique(subbasin_types)
base_pas = load("src/distributed/cache/hbv_opt.jld2", "opt_params")

# generate multi-types parameters and initstates
node_params = NamedTuple{Tuple(subtype_category)}(
    repeat([ComponentVector(base_pas[:params]; rapid_k=0.8, rapid_x=0.5)], length(subtype_category))
)
convert_params = NamedTuple{Tuple(hybas_id)}([(subarea=sa,) for sa in subbasin_areas])
combine_params = merge_recursive(node_params, convert_params)
node_initstates = NamedTuple{Tuple(subtype_category)}(repeat([base_pas[:initstates]], length(subtype_category)))
node_pas = ComponentVector(params=combine_params, initstates=node_initstates)
# prepare input data
input = map(names(subbasin_prcp_df)[2:end]) do name
    (prcp=subbasin_prcp_df[!, name], temp=subbasin_temp_df[!, name], pet=subbasin_pet_df[!, name])
end

configs = [
    (ptypes=subbasin_types, solver=ODESolver()),
    (ptypes=subbasin_types, stypes=subbasin_types, timeidx=time_idx, solver=ODESolver()),
    (ptypes=subbasin_types, stypes=subbasin_types, timeidx=time_idx, solver=ODESolver()),
    (ptypes=subbasin_types, stypes=subbasin_types, timeidx=time_idx, solver=ODESolver()),
    (ptypes=hybas_id, timeidx=time_idx, solver=ODESolver()),
    (ptypes=subbasin_types, timeidx=time_idx, solver=DiscreteSolver()),
]

hbv_rapid_output = ioadapter(input, node_pas, config=configs)

# outlet_wrapper = HydroModels.ComputeComponentOutlet(hbv_rapid, 1)
# hbv_rapid_opt = GradOptimizer(component=outlet_wrapper, maxiters=100, adtype=AutoForwardDiff(), solve_alg=Adam(1e-2))
# tunable_pas = ComponentVector(params=node_params)
# const_pas = ComponentVector(initstates=node_initstates, params=convert_params)
# opt_params, loss_df = hbv_rapid_opt(
#     [input],
#     [(flow=flow_vec,)],
#     tunable_pas=tunable_pas,
#     const_pas=const_pas,
#     config=[configs],
#     return_loss_df=true
# )
# hbv_rapid_output = hbv_rapid(input, node_pas, convert_to_ntp=true, config=configs)
# plot(hbv_rapid_output[1].q[1:1000], label="q",)
# plot!(hbv_rapid_output[1].flow[1:1000], label="flow",)


plot(hbv_rapid_output[1].flow_routed, 
label="simulated",
xlabel="time index",
ylabel="discharge (m³/s)",
xticks=(0:2000:length(time_idx)),  # Set x ticks interval to 2000
background=:white, 
fontfamily="calibri",
formatter=:plain,
dpi=300,
alpha=0.7,  # Set transparency
)
plot!(flow_vec, label="observed", alpha=0.7)
savefig("src/distributed/rapid_hbv.png")