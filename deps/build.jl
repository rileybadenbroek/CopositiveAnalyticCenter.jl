if haskey(ENV, "GITHUB_ACTIONS")
    println("Looks like a GitHub action. I'm skipping installation to prevent " *
    "errors saying Gurobi is not installed.")
    exit(0)
end
