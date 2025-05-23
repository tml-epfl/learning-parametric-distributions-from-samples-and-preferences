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
            default = "error1d"
        "--instance"
            help = "Instance considered."
            arg_type = String
            default = "Rayleigh"
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
            default = 10
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
@everywhere function runit(seed, N, instance, μmin, μmax, batch, sizemax)
	rng = MersenneTwister(seed);

	if instance == "Rayleigh"
		# Bayesian instance
		σ2 = (μmax - μmin) * rand(rng) + μmin;
	    dist = Rayleigh(sqrt(σ2));

	    # Observations
	    data = rand(rng, dist, (2, N));
	    X = data[1, :];
	    Y = data[2, :];
        logratioprob = logpdf(dist, X) .- logpdf(dist, Y);
    	Z = 2 * (logratioprob .> 0) .- 1;
        Zsto = 2 * (σ.(logratioprob) .< rand(rng, N)) .- 1;

        # Computations
        X2 = X.^2; 
        Y2 = Y.^2; 
        D = log.(X ./ Y);
        S = (X2 .- Y2) / 2;

    	# SO MLE
        SOMLEs = (cumsum(X2) + cumsum(Y2)) ./ (4 * collect(1:N));

        # Worst-case estimator
        WEs = zeros(sizemax);
        model1 = JuMP.Model(Ipopt.Optimizer);
        set_silent(model1);
        @variable(model1, x1);
        @objective(model1, Max, abs(x1 - σ2));
        for k in 1:N
            @constraint(model1, Z[k] * (x1 * D[k] - S[k]) >= 0);
            # Objective
            if k % batch == 0
                iloc = Int(k / batch);
                optimize!(model1);
                WEs[iloc] = value(x1);
            end
        end
        model1 = nothing;

        # Any estimator
        AEs = zeros(sizemax);
        model2 = JuMP.Model(Ipopt.Optimizer);
        set_silent(model2);
        @variable(model2, x2);
        @objective(model2, Min, 0.);
        for k in 1:N
            @constraint(model2, Z[k] * (x2 * D[k] - S[k]) >= 0);
            # Objective
            if k % batch == 0
                iloc = Int(k / batch);
                optimize!(model2);
                AEs[iloc] = value(x2);
            end
        end
        model2 = nothing;

        return σ2, SOMLEs .- σ2, WEs .- σ2, AEs .- σ2;#, DPMLEs .- σ2, LLEs .- σ2, SPMLEs .- σ2;
	else
		@error "Not Implemented";
	end
end
    

# Run the experiments in parallel
@time data = pmap(
    (i,) -> runit(seed + i, Nsteps, instance, μmin, μmax, batch, sizemax),
    1:Nruns
);


# Save everything using JLD2.
@save "$(experiment_dir)$(experiment_name).dat" data Nruns Nsteps instance μmin μmax batch sizemax;

