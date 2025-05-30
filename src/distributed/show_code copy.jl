#* build UH-based Model
@variables q q_routed
@parameters lag

uh_2 = @unithydro begin
    uh_func = begin
        2MAXBAS => (1 - 0.5 * (2 - t / MAXBAS)^2.5)
        MAXBAS => (0.5 * (t / MAXBAS)^2.5)
    end
    uh_vars = [Q1]
    configs = (solvetype=:SPARSE, suffix=:_routed)
end
dpl_hbv_uh_ = HydroModel(
    name=:rapid_hbv,
    components=[dpl_hbv_model.components..., uh1]
)
dpl_hbv_uh = HydroWrapper(
    dpl_hbv_uh_,
    pre=(i, p; kw...) -> est_func1(i, p; kw...),
    post=(output) -> sum(output, dims=2)
)

maxbas = 3.5

function tmp(t)
    println(t)
    if t > maxbas
        return (1 - 0.5 * (2 - t / maxbas)^2.5)
    else
        return (0.5 * (t / maxbas)^2.5)
    end
end

output = vcat([0], tmp.(1:1:2*maxbas))
fig = bar(output[2:end] .- output[1:end-1],
    xlabel="Time", ylabel="Weight",
    fontsize=18, fontfamily="Times", color=Colors.JULIA_LOGO_COLORS.green,
    legend=false, xtickfontsize=16, ytickfontsize=16, 
    xlabelfontsize=18, ylabelfontsize=18,
    grid=false, linewidth=1, dpi=300)
savefig(fig, "maxbas_weight.png")
#* build Grid-Based Model
@variables s_river q_routed
@parameters lag

grid_basin_info = load("data/hanjiang/grid_basin_info.jld2")
flwdir_matrix = grid_basin_info["flwdir_matrix"]
index_info = collect.(grid_basin_info["index_info"])

grid_route = @hydroroute begin
    fluxes = begin
        @hydroflux q_routed ~ s_river / (1 + lag)
    end
    dfluxes = begin
        @stateflux s_river ~ q - q_routed
    end
    aggr_func = build_aggr_func(flwdir_matrix, index_info)
    # aggr_func = build_aggr_func(netowrk)
end
dpl_hbv_grid_ = HydroModel(
    name=:grid_hbv,
    components=[dpl_hbv_model.components..., grid_route]
)
dpl_hbv_grid = HydroWrapper(
    dpl_hbv_grid_,
    pre=(i, p; kw...) -> est_func2(i, p; kw...)
)

#* build Rapid-Based Model
@variables q flow flow_routed
@parameters subarea
vector_basin_info = load("tutorials/distributed/data/vector_basin_info.jld2")
vector_dgraph = vector_basin_info["dg8"]
subbasin_areas = vector_basin_info["subbasin_area"]
subbasin_rivlen = vector_basin_info["riv_len_df"]
hybas_id = [Symbol("subbasin_$i") for i in vector_basin_info["hybas_id"]]
flow_convert_flux = HydroFlux([q] => [flow], [subarea], exprs=[q * 24 * 3600 / (subarea * 1e6) * 1e3])
rapid_route = RapidRoute([flow] => [flow_routed], network=vector_dgraph)
dpl_hbv_rapid = HydroModel(name=:rapid_hbv, components=[dpl_hbv_model.components..., flow_convert_flux, rapid_route])

