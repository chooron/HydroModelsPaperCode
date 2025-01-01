using Random, KolmogorovArnold
using HydroModels
using Lux
using Lux:LuxCore.Internal
using ComponentArrays
rng = Random.default_rng()

LuxCore.Internal.get_empty_state(l::KDense) = (;grid = collect(LinRange(l.grid_lims..., l.grid_len)))

@variables a b c
in_dim,hid_dim, out_dim, grid_len = 2, 5, 1, 4
chain = Lux.Chain(
    KDense(in_dim, hid_dim, grid_len),
    KDense(hid_dim, out_dim, grid_len),
    name=:chain
)
flux = NeuralFlux([a,b]=>[c], chain)
ps,st = Lux.setup(rng, chain)
nn_pas = ComponentVector(nn=(chain=Vector(ComponentVector(ps)),))
flux([1.0, 2.0], nn_pas)