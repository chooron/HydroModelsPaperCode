using SymbolicUtils
using SymbolicUtils.Code
using Symbolics
using BenchmarkTools
using Zygote
using ProfileView

@variables a b c d
@variables p1 p2 p3

assign_list = [
    Assignment(c, a * p1 + b * p2),
    Assignment(d, c * p1 + b * p3),
]

flux_output_array = MakeArray([c, d], Vector)

func1 = Func([DestructuredArgs([a,b]), DestructuredArgs([p1, p2, p3])], [], Let(assign_list, flux_output_array, false))
test_func1 = eval(toexpr(func1))
test_func2(i,p) = begin
    a = i[1]
    b = i[2]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]
    c = a * p1 + b * p2
    d = c * p1 + b * p3
    [c, d]
end

# @btime test_func1([2,3],[2,3,4])
# @btime test_func2([2,3],[2,3,4])
# # 42.525 ns (6 allocations: 240 bytes)
# # 26.004 ns (4 allocations: 176 bytes)
# ProfileView.@profview gradient((p)->sum(test_func1([2,3], p)), [2,3,4])
# ProfileView.@profview gradient((p)->sum(test_func2([2,3], p)), [2,3,4])
# # 4.343 μs (93 allocations: 5.38 KiB)
# # 74.486 ns (11 allocations: 416 bytes)

# input = ones(2,10000)
# params = [2,3,4]
# @btime test_func1.(eachslice(input,dims=2), Ref(params));
# @btime test_func2.(eachslice(input,dims=2), Ref(params));
# # 152.900 μs (20008 allocations: 859.56 KiB)
# # 162.300 μs (20021 allocations: 860.09 KiB)

# gradient((p)->sum(sum(test_func1.(eachslice(input,dims=2), Ref(p)))), [2,3,4])
# gradient((p)->sum(sum(test_func2.(eachslice(input,dims=2), Ref(p)))), [2,3,4])
# 29.549 ms (870129 allocations: 26.86 MiB)
# 1.903 ms (100106 allocations: 7.33 MiB)

for (i,x,y) in enumerate(zip([1,1,1],[2,3,4]))
    println((i,x,y))
end