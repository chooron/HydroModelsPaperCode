using Plots, JLD2
using Colors

"""
    plot_flowdir_matrix(flwdir_matrix::Matrix{Int}; 
                         cmap=:viridis, 
                         fontsize=6, 
                         figsize=(800, 600))

绘制流向矩阵的热力图并在每个单元格上标注对应的数值。

参数:
- `flwdir_matrix::Matrix{Int}`: 流向矩阵
- `cmap::Symbol`: 颜色映射，默认为`:viridis`
- `fontsize::Int`: 标注文本的字体大小，默认为6
- `figsize::Tuple{Int,Int}`: 图像大小，默认为(800, 600)

返回:
- Plots.Plot 对象
"""
function plot_flowdir_matrix(flwdir_matrix::Matrix{Int}; 
                             cmap=:viridis, 
                             fontsize=6, 
                             figsize=(800, 600))
    # 创建掩码矩阵，将-9999替换为NaN以便在热力图中显示为透明
    masked_matrix = replace(float.(flwdir_matrix), -9999.0 => NaN)
    
    # 绘制热力图
    p = heatmap(masked_matrix, 
                c=cmap, 
                aspect_ratio=:equal, 
                size=figsize,
                colorbar=true, 
                title="流向矩阵热力图",
                xlabel="列索引", 
                ylabel="行索引",
                xflip=false,  # 保持x轴方向不变
                yflip=true)   # 反转y轴，使左上角为(1,1)
    
    # 在每个单元格上标注数值
    for i in 1:size(flwdir_matrix, 1)
        for j in 1:size(flwdir_matrix, 2)
            val = flwdir_matrix[i, j]
            if val != -9999
                # 计算文本颜色：对于深色背景使用白色文本，对于浅色背景使用黑色文本
                if !isnan(masked_matrix[i, j])
                    # 获取背景颜色的亮度来决定文本颜色
                    annotate!(j, i, text(string(val), fontsize, :center, 
                                         val > 64 ? :white : :black))
                end
            end
        end
    end
    
    return p
end

"""
    plot_flowdir_from_file(filepath::String)

从JLD2文件加载流向矩阵并绘制热力图。

参数:
- `filepath::String`: JLD2文件路径

返回:
- Plots.Plot 对象
"""
function plot_flowdir_from_file(filepath::String)
    data = load(filepath)
    flwdir_matrix = data["flwdir_matrix"]
    
    return plot_flowdir_matrix(flwdir_matrix)
end

# 示例用法
if abspath(PROGRAM_FILE) == @__FILE__
    # 直接从文件加载并绘制
    p = plot_flowdir_from_file("data/hanjiang/grid_basin_info.jld2")
    display(p)
    savefig(p, "flowdir_matrix.png")
    
    # 或者，如果已有流向矩阵变量，可以直接使用
    # grid_basin_info = load("data/hanjiang/grid_basin_info.jld2")
    # flwdir_matrix = grid_basin_info["flwdir_matrix"]
    # p = plot_flowdir_matrix(flwdir_matrix)
    # display(p)
    # savefig(p, "flowdir_matrix.png")
end 