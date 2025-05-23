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
            default = "Laplace"
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

	if instance == "Laplace"
		# Bayesian instance
		μ = (μmax - μmin) * rand(rng) + μmin;
	    dist = Laplace(μ);

	    # Observations
	    data = rand(rng, dist, (2, N));
	    X = data[1, :];
	    Y = data[2, :];
        logratioprob = logpdf(dist, X) .- logpdf(dist, Y);
    	Z = 2 * (logratioprob .> 0) .- 1;
        Zsto = 2 * (σ.(logratioprob) .< rand(rng, N)) .- 1;

    	# SO MLE
        SOMLEs = zeros(sizemax);
        for i in 1:sizemax
            Nloc = Int(i * batch);
            SOMLEs[i] = median(data[:, 1:Nloc]);
        end

        # Worst-case estimator
        WEs = zeros(sizemax);
        model1 = JuMP.Model(Ipopt.Optimizer);
        set_silent(model1);
        @variable(model1, x1);
        @objective(model1, Max, abs(x1 - μ));
        for k in 1:N
            @constraint(model1, Z[k] * (abs(Y[k] - x1) - abs(X[k] - x1)) >= 0);
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
            @constraint(model2, Z[k] * (abs(Y[k] - x2) - abs(X[k] - x2)) >= 0);
            # Objective
            if k % batch == 0
                iloc = Int(k / batch);
                optimize!(model2);
                AEs[iloc] = value(x2);
            end
        end
        model2 = nothing;

        # DP MLE
        DPMLEs = zeros(sizemax);
        for i in 1:sizemax
            Nloc = Int(i * batch);

            model3 = JuMP.Model(Ipopt.Optimizer);
            set_silent(model3);
            @variable(model3, x3);
            @objective(model3, Min, sum(-abs.(Y[1:Nloc] .- x3) - abs.(X[1:Nloc] .- x3)));
            for k in 1:Nloc
                @constraint(model3, Z[k] * (abs(Y[k] - x3) - abs(X[k] - x3)) >= 0);
            end
            optimize!(model3);
            DPMLEs[i] = value(x3);
        end

		# LLE     
        LLEs = zeros(sizemax);
        for i in 1:sizemax
            Nloc = Int(i * batch);

            model4 = JuMP.Model(Ipopt.Optimizer);
            set_silent(model4);
            @variable(model4, x4);
            @objective(model4, Min, sum(log.(1 .+ exp.(-Zsto[1:Nloc] .* (abs.(Y[1:Nloc] .- x4) .- abs.(X[1:Nloc] .- x4))))) / Nloc - 
                                    sum(abs.(Y[1:Nloc] .- x4) + abs.(X[1:Nloc] .- x4)) / Nloc);
            optimize!(model4);
            LLEs[i] = value(x4);
        end

        # SP MLE
        SPMLEs = zeros(sizemax);
        for i in 1:sizemax
            Nloc = Int(i * batch);

            model5 = JuMP.Model(Ipopt.Optimizer);
            set_silent(model5);
            @variable(model5, x5);
            @objective(model5, Min, sum(log.(1 .+ exp.(-Z[1:Nloc] .* (abs.(Y[1:Nloc] .- x5) .- abs.(X[1:Nloc] .- x5))))) / Nloc - 
                                    sum(abs.(Y[1:Nloc] .- x5) + abs.(X[1:Nloc] .- x5)) / Nloc);
            optimize!(model5);
            SPMLEs[i] = value(x5);
        end

        return μ, SOMLEs .- μ, WEs .- μ, AEs .- μ, DPMLEs .- μ, LLEs .- μ, SPMLEs .- μ;
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

