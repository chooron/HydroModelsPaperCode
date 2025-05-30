using Plots, JLD2

"""
    绘制流向矩阵热力图并标注数值

此脚本直接从JLD2文件加载流向矩阵并绘制热力图，
标注每个单元格的数值。无效值(-9999)将被设置为透明。
"""

function main()
    # 从JLD2文件加载数据
    grid_basin_info = load("data/hanjiang/grid_basin_info.jld2")
    flwdir_matrix = grid_basin_info["flwdir_matrix"]
    
    # 创建掩码矩阵，将-9999替换为NaN以便在热力图中显示为透明
    masked_matrix = replace(float.(flwdir_matrix), -9999.0 => NaN)
    
    # 设置绘图参数
    fontsize = 8
    figsize = (800, 600)
    
    # 绘制热力图
    p = heatmap(masked_matrix, 
                c=:viridis, 
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
                # 根据数值大小决定文本颜色
                text_color = val > 64 ? :white : :black
                annotate!(j, i, text(string(val), fontsize, :center, text_color))
            end
        end
    end
    
    # 保存图像
    savefig(p, "flowdir_matrix.png")
    println("已生成流向矩阵热力图并保存为 flowdir_matrix.png")
    
    # 显示图像
    display(p)
    
    return p
end

main()