using Plots
include("plot_flowdir.jl")

# 示例：直接使用 optimal_grid_hbv.jl 中已经加载的流向矩阵
include("optimal_grid_hbv.jl")  # 这会加载 flwdir_matrix 变量

# 使用不同的颜色方案
p1 = plot_flowdir_matrix(flwdir_matrix, cmap=:viridis, fontsize=8)
savefig(p1, "flowdir_matrix_viridis.png")

# 使用不同的颜色方案
p2 = plot_flowdir_matrix(flwdir_matrix, cmap=:plasma, fontsize=8)
savefig(p2, "flowdir_matrix_plasma.png")

# 更适合流向编码的离散色彩图
p3 = plot_flowdir_matrix(flwdir_matrix, cmap=:Set1, fontsize=8)
savefig(p3, "flowdir_matrix_discrete.png")

println("已生成流向矩阵热力图并保存为PNG文件") 