@everywhere using ArgParse, JLD2, Printf, JSON, Dates, IterTools, Random;
@everywhere using LinearAlgebra, Statistics, Distributions, JuMP;
@everywhere import Ipopt, HiGHS;
using Distributed;

function parse_commandline()
    s = ArgParseSettings();

    @add_arg_table! s begin
        "--save_dir"
            help = "Directory for saving the experiment's data."
            arg_type = String
            default = "experiments/"
        "--data_dir"
            help = "Directory for loading the data."
            arg_type = String
            default = "data/"
        "--seed"
            help = "Seed."
            arg_type = Int64
            default = 42
        "--expe"
            help = "Experiment considered."
            arg_type = String
            default = "betterEstimVard"
        "--instance"
            help = "Instance considered."
            arg_type = String
            default = "Gaussian"
        "--Nsteps"
            help = "Number of steps."
            arg_type = Int64
            default = 1000
        "--Nruns"
            help = "Number of runs of the experiment."
            arg_type = Int64
            default = 4
        "--dimMax"
            help = "Max dimension."
            arg_type = Int64
            default = 8
        "--dimStep"
            help = "Step between dimensions."
            arg_type = Int64
            default = 2
    end

    parse_args(s);
end

# Parameters
parsed_args = parse_commandline();
save_dir = parsed_args["save_dir"];
data_dir = parsed_args["data_dir"];
seed = parsed_args["seed"];
instance = parsed_args["instance"];
expe = parsed_args["expe"];
Nsteps = parsed_args["Nsteps"];
Nruns = parsed_args["Nruns"];
dimMax = parsed_args["dimMax"];
dimStep = parsed_args["dimStep"];

# Naming files and folder
now_str = Dates.format(now(), "dd-mm_HHhMM");
experiment_name = "exp_" * expe * "_inst_" * instance * "_dMax_" * string(dimMax) * "_dStep_" * string(dimStep) * "_n_" * string(Nsteps) * "_N_" * string(Nruns);
experiment_dir = save_dir * now_str * ":" * experiment_name * "/";
mkdir(experiment_dir);
open("$(experiment_dir)parsed_args.json","w") do f
    JSON.print(f, parsed_args)
end


@everywhere function chebyshev_center(A, b)
    m, n = size(A)  # Number of constraints (m) and variables (n)
    
    model = Model(HiGHS.Optimizer)  # Use HiGHS as the solver
    set_silent(model);
    
    @variable(model, x[1:n])  # Decision variables for the center of the ball
    @variable(model, r >= 0)  # Radius (must be non-negative)
    
    # Normalize constraint rows to ensure the ball is properly inscribed
    norms = [norm(A[i, :]) for i in 1:m]
    
    # Chebyshev center constraints
    @constraint(model, [i = 1:m], A[i, :] ⋅ x + r * norms[i] <= b[i])
    
    # Objective: Maximize the radius
    @objective(model, Max, r)
    
    # Solve the optimization problem
    optimize!(model)
    
    return value.(x), value(r)  # Return the center and radius
end


@everywhere σ(x) = 1/(1 + exp(-x));
μmin = 1;
μmax = 2;
@everywhere function runit(seed, d, N, instance, μmin, μmax)
	rng = MersenneTwister(seed);

	if instance == "Gaussian"
		# Bayesian instance
        μ = (μmax - μmin) * rand(rng, d) .+ μmin;
	    dist = Normal();

	    # Observations
        data = rand(rng, dist, (2, N, d));
        X = data[1, :, :] .+ μ';
        Y = data[2, :, :] .+ μ';
        S = (X .+ Y) / 2;
        D = X .- Y;
        B = sum(D .* S, dims=2);
        logratioprob = sum(D .* (μ' .- S), dims=2)
        Z = 2 * (logratioprob .> 0) .- 1;
        Zsto = 2 * (σ.(logratioprob) .< rand(rng, N)) .- 1;

        # SO MLE
        SOMLEs = mean(S, dims=1)[1,:];

        # Any estimator
        model2 = JuMP.Model(HiGHS.Optimizer);
        set_silent(model2);
        @variable(model2, x2[u in 1:d]);
        @objective(model2, Min, 0);
        for k in 1:N
            @constraint(model2, Z[k] * (dot(D[k,:], x2) - B[k]) >= 0);
        end
        optimize!(model2);
        AEs = copy(value.(x2));
        model2 = nothing;

        # DP MLE
        model3 = JuMP.Model(Ipopt.Optimizer);
        set_silent(model3);
        @variable(model3, x3[u in 1:d]);
        @objective(model3, Min, sum((x3 .- SOMLEs).^2));
        for k in 1:N
            @constraint(model3, Z[k] * (dot(D[k,:], x3) - B[k]) >= 0);
        end
        optimize!(model3);
        DPMLEs = copy(value.(x3));
        model3 = nothing;

        # Constraints
        A = - Z .* D;
        b = - Z .* B;
            
        # Chebyshev Center
        _cen, _rad = chebyshev_center(A, b);
    
        # Chebyshev Center Estimator
        CCEs = _cen;

        return μ, SOMLEs .- μ, AEs .- μ, DPMLEs .- μ, CCEs .- μ;
	else
		@error "Not Implemented";
	end
end

rangeDims = collect(1:dimStep:dimMax) .+ 1;
# Run the experiments in parallel
@time data = pmap(
    ((dimension, i),) -> runit(seed + i, dimension, Nsteps, instance, μmin, μmax),
    Iterators.product(rangeDims, 1:Nruns)
);


# Save everything using JLD2.
@save "$(experiment_dir)$(experiment_name).dat" data Nruns dimMax dimStep Nsteps instance μmin μmax;

