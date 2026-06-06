# The Std* distributions are the canonical "spaces" we transport into:
# zero-mean / unit-scale references of arbitrary shape. `_sampling.jl` must
# come first since `std_tdist.jl` / `std_inversegamma.jl` use `_rand_gamma`.

include("interface.jl")
include("misc.jl")
include("std_normal.jl")
include("std_uniform.jl")
include("std_exponential.jl")
include("std_tdist.jl")
include("std_inversegamma.jl")
