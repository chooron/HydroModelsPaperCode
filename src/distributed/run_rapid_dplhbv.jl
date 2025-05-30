using CSV
using DataFrames
using ComponentArrays
using BenchmarkTools
using NamedTupleTools
using Optimization
using OptimizationOptimisers
using JLD2
using Plots
using HydroModels
include("../models/dplHBV.jl")


TOTAL_AREA = 15230.3
convert_eff = 24 * 3600 / (TOTAL_AREA * 1e6) * 1e3
# build model based on HBV and rapid module
@variables q flow flow_routed
@parameters subarea
vector_basin_info = load("tutorials/distributed/data/vector_basin_info.jld2")
vector_dgraph = vector_basin_info["dg8"]
subbasin_areas = vector_basin_info["subbasin_area"]
subbasin_rivlen = vector_basin_info["riv_len_df"]
subbasin_id = vector_basin_info["hybas_id"]
hybas_id = [Symbol("subbasin_$i") for i in subbasin_id]
flow_convert_flux = HydroFlux([q] => [flow], [subarea], exprs=[q * 24 * 3600 / (subarea * 1e6) * 1e3])
rapid_route = RapidRoute([flow] => [flow_routed], network=vector_dgraph)
hbv_rapid = HydroModel(name=:rapid_hbv, components=[dpl_hbv_model.components..., flow_convert_flux, rapid_route])

# load data
select_idx = 2000:2400
time_idx = collect(1:length(select_idx))
vector_meto = load("tutorials/distributed/data/vector_meto.jld2")
flow_df = CSV.read("tutorials/distributed/data/flow.csv", DataFrame)
flow_vec = flow_df[select_idx, "FLOW"]
subbasin_pet_df = vector_meto["pet"][select_idx, :]
subbasin_prcp_df = vector_meto["prec"][select_idx, :]
subbasin_temp_df = vector_meto["temp"][select_idx, :]
subbasin_type_ids = CSV.read("tutorials/distributed/data/subbasin_types.csv", DataFrame)[!, :SUBBASIN_TYPE]
subbasin_type_category = unique(subbasin_type_ids)
subbasin_type_index = Int.(indexin(subbasin_type_ids, subbasin_type_category))
# 移除或注释掉原来的subbasin_types定义
# subbasin_types = [Symbol("subtype_$i") for i in subbasin_type_ids]
subbasin_areas = load("tutorials/distributed/data/vector_basin_info.jld2")["subbasin_area"]
subtype_category = unique(subbasin_type_ids)
base_pas = load("tutorials/distributed/cache/hbv_opt.jld2", "opt_params")

#* generate multi-types parameters and initstates
psnn_ps, psnn_st = Lux.setup(StableRNG(123), params_nn)
psnn_ps_vec = Vector(ComponentVector(psnn_ps))
merge_params = ComponentVector(base_pas[:params]; k=1.0, x=0.4)
node_params = ComponentVector(params=NamedTuple{keys(merge_params)}(
    [repeat([merge_params[k]], length(subtype_category)) for k in keys(merge_params)]
))
convert_params = ComponentVector(subarea=subbasin_areas)
combine_params = ComponentVector(params=merge(node_params, convert_params))
merge_initstates = ComponentVector(base_pas[:initstates])
node_initstates = ComponentVector(NamedTuple{keys(merge_initstates)}(
    [repeat([merge_initstates[k]], length(subtype_category)) for k in keys(merge_initstates)]
))
node_pas = ComponentVector(params=combine_params, initstates=node_initstates, nn=(pnn=psnn_ps_vec,))
#* prepare input data
input = map(names(subbasin_prcp_df)[2:end]) do name
    (prcp=subbasin_prcp_df[!, name], temp=subbasin_temp_df[!, name], pet=subbasin_pet_df[!, name])
end
input_arr = stack([stack(i, dims=1) for i in input], dims=2)
#* prepare configs for each component
configs = [
    (ptyidx=subbasin_type_index, solver=ODESolver()),
    (ptyidx=subbasin_type_index, styidx=subbasin_type_index, timeidx=time_idx, solver=ODESolver()),
    (ptyidx=subbasin_type_index, styidx=subbasin_type_index, timeidx=time_idx, solver=ODESolver()),
    (ptyidx=subbasin_type_index, styidx=subbasin_type_index, timeidx=time_idx, solver=ODESolver()),
    (ptyidx=collect(1:length(subbasin_id)), timeidx=time_idx, solver=ODESolver()),
    (ptyidx=subbasin_type_index, timeidx=time_idx, solver=HydroModels.DiscreteSolver()),
]
#* run model
hbv_rapid_output = hbv_rapid(input_arr, node_pas, convert_to_ntp=true, config=configs)


# outlet_wrapper = HydroModels.ComputeComponentOutlet(hbv_rapid, 1)
# hbv_rapid_opt = GradOptimizer(component=outlet_wrapper, maxiters=100, adtype=AutoZygote(), solve_alg=Adam(1e-2))
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
# plot(loss_df.loss)