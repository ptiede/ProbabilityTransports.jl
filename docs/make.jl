using ProbabilityTransports
using Documenter

DocMeta.setdocmeta!(ProbabilityTransports, :DocTestSetup, :(using ProbabilityTransports); recursive=true)

makedocs(;
    modules=[ProbabilityTransports],
    authors="Paul Tiede <ptiede91@gmail.com> and contributors",
    repo="https://github.com/ptiede/ProbabilityTransports.jl/blob/{commit}{path}#{line}",
    sitename="ProbabilityTransports.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ptiede.github.io/ProbabilityTransports.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ptiede/ProbabilityTransports.jl",
    devbranch="main",
)
