try
    import Gurobi
    open(joinpath(@__DIR__, "USESOLVER"), "w") do io
        write(io, "Gurobi")
    end
catch
    import ECOS, Cbc
    open(joinpath(@__DIR__, "USESOLVER"), "w") do io
        write(io, "Free")
    end
end
