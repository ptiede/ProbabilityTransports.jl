module ProbabilityTransports

# Write your package code here.

abstract type Transport end

struct BoundaryTransport{S,E,T}
    start::S
    stop::E
    transform::T
end


struct TVTransport{E, T}
    stop::E
    transform::T
end


function transform_and_logdensity(t::TVTransport, x)
    y, lj = TV.transform_and_logjac(t, x)
    return logpdf(d, y) + lj
end

end
