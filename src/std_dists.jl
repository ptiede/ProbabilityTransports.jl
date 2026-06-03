# The Std* distributions are the canonical "spaces" we transport into:
# zero-mean / unit-scale references of arbitrary shape. `_sampling.jl` must
# come first since `std_tdist.jl` / `std_inversegamma.jl` use `_rand_gamma`.

include("StdDists/_sampling.jl")
include("StdDists/std_normal.jl")
include("StdDists/std_uniform.jl")
include("StdDists/std_exponential.jl")
include("StdDists/std_tdist.jl")
include("StdDists/std_inversegamma.jl")
