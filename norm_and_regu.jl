@everywhere using ArgParse, JLD2, Printf, JSON, Dates, IterTools, Random;
@everywhere using LinearAlgebra, Statistics, Distributions, JuMP;
@everywhere import Ipopt;
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
            default = "normregu"
        "--instance"
            help = "Instance considered."
            arg_type = String
            default = "Gaussian"
        "--Nsteps"
            help = "Number of steps."
            arg_type = Int64
            default = 100
        "--Nruns"
            help = "Number of runs of the experiment."
            arg_type = Int64
            default = 20
        "--batch"
            help = "Size of batches."
            arg_type = Int64
            default = 10
        "--sizemax"
            help = "Number of Optimization by solver."
            arg_type = Int64
            default = 100
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
batch = parsed_args["batch"];
sizemax = parsed_args["sizemax"];

# Naming files and folder
now_str = Dates.format(now(), "dd-mm_HHhMM");
experiment_name = "exp_" * expe * "_inst_" * instance * "_n_" * string(Nsteps) * "_N_" * string(Nruns);
experiment_dir = save_dir * now_str * ":" * experiment_name * "/";
mkdir(experiment_dir);
open("$(experiment_dir)parsed_args.json","w") do f
    JSON.print(f, parsed_args)
end

@everywhere σ(x) = 1/(1 + exp(-x));
μmin = 1;
μmax = 2;
@everywhere function runit(seed, N, instance, μmin, μmax, batch, sizemax, βs, λs)
	rng = MersenneTwister(seed);

	if instance == "Gaussian"
		# Bayesian instance
		μ = (μmax - μmin) * rand(rng) + μmin;
	    dist = Normal(μ);

	    # Observations
	    data = rand(rng, dist, (2, N));
	    X = data[1, :];
	    Y = data[2, :];
        logratioprob = logpdf(dist, X) .- logpdf(dist, Y);
    	Z = 2 * (logratioprob .> 0) .- 1;
        Zsto = 2 * (σ.(logratioprob) .< rand(rng, N)) .- 1;

    	# Intermediate computations
        S = (X .+ Y) / 2;
        minS = minimum(S);
        maxS = maximum(S);
        M = Z .* (X .- Y) .< 0;
        SU = S .* M .+ maxS * (.!M);
        SL = S .* (.!M) .+ minS * M;
        U = accumulate(min, SU);
        L = accumulate(max, SL);

    	# SO MLE
    	SOMLEs = (cumsum(X) + cumsum(Y)) ./ (2 * collect(1:N));

        # DP MLE
        DPMLEs = clamp.(SOMLEs, L, U);

		# SP estimator deterministic     
        SPEβs = zeros((size(βs)[1], sizemax));
        SPEλs = zeros((size(λs)[1], sizemax));
        for i in 1:sizemax
            Nloc = Int(i * batch);

            # Intermediate computations
            V1 = Z[1:Nloc] .* (X[1:Nloc] .- Y[1:Nloc]);
            V2 = Z[1:Nloc] .* (X[1:Nloc] .- Y[1:Nloc]) .* S[1:Nloc];

            for (k, β) in enumerate(βs)
                # Definition model
                model1 = JuMP.Model(Ipopt.Optimizer);
                set_silent(model1);
                @variable(model1, x1);

                @objective(model1, Min, (x1 - SOMLEs[Nloc])^2 + 
                                        sum(log.(1 .+ exp.(β * (V2 .- x1 * V1)))) / Nloc);
                
                # Solve
                optimize!(model1);
                SPEβs[k, i] = value(x1);

                model1 = nothing;
            end

            for (k, λ) in enumerate(λs)
                # Definition model
                model1 = JuMP.Model(Ipopt.Optimizer);
                set_silent(model1);
                @variable(model1, x1);

                @objective(model1, Min, (x1 - SOMLEs[Nloc])^2 + 
                                        λ * sum(log.(1 .+ exp.(V2 .- x1 * V1))) / Nloc);
                
                # Solve
                optimize!(model1);
                SPEλs[k,i] = value(x1);

                model1 = nothing;
            end
        end

        return μ, SOMLEs .- μ, DPMLEs .- μ, SPEλs .- μ, SPEβs .- μ;
	else
		@error "Not Implemented";
	end
end
    
βs = [0.01, 0.1, 1., 10., 100.];
λs = [0.01, 0.1, 1., 10., 100.];

# Run the experiments in parallel
@time data = pmap(
    (i,) -> runit(seed + i, Nsteps, instance, μmin, μmax, batch, sizemax, βs, λs),
    1:Nruns
);


# Save everything using JLD2.
@save "$(experiment_dir)$(experiment_name).dat" data Nruns Nsteps instance μmin μmax batch sizemax βs λs;

