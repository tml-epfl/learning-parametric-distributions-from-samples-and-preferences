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
            default = "errorlarged"
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
            default = 4
        "--batch"
            help = "Size of batches."
            arg_type = Int64
            default = 10
        "--dimension"
            help = "Dimensionality."
            arg_type = Int64
            default = 2
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
dimension = parsed_args["dimension"];

# Naming files and folder
now_str = Dates.format(now(), "dd-mm_HHhMM");
experiment_name = "exp_" * expe * "_inst_" * instance * "_d_" * string(dimension) * "_n_" * string(Nsteps) * "_N_" * string(Nruns);
experiment_dir = save_dir * now_str * ":" * experiment_name * "/";
mkdir(experiment_dir);
open("$(experiment_dir)parsed_args.json","w") do f
    JSON.print(f, parsed_args)
end

@everywhere σ(x) = 1/(1 + exp(-x));
μmin = 1;
μmax = 2;
@everywhere function runit(seed, d, N, instance, μmin, μmax, batch, sizemax)
	rng = MersenneTwister(seed);
    #size = Int(N / batch);
    # JuMP gets StackOverflowError when checking that this is a convex program for too large size


	if instance == "Gaussian"
		# Bayesian instance
        μ = (μmax - μmin) * rand(rng, d) .+ μmin;
	    dist = Normal();

	    # Observations
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
        SOMLEs = zeros((sizemax, d));
        for i in 1:sizemax
            Nloc = Int(i * batch);

            SOMLEs[i,:] = mean(S[1:Nloc,:], dims=1);
        end
        
        # Worst-case estimator
        WEs = zeros((sizemax, d));
        model1 = JuMP.Model(Ipopt.Optimizer);
        set_silent(model1);
        @variable(model1, x1[u in 1:d]);
        @objective(model1, Max, sum(abs.(x1 .- μ)));
        for k in 1:N
            @constraint(model1, Z[k] * (dot(D[k,:], x1) - B[k]) >= 0);
            # Objective
            if k % batch == 0
                iloc = Int(k / batch);
                optimize!(model1);
                WEs[iloc, :] = copy(value.(x1));
            end
        end
        model1 = nothing;

        # Any estimator
        AEs = zeros((sizemax, d));
        model2 = JuMP.Model(HiGHS.Optimizer);
        set_silent(model2);
        @variable(model2, x2[u in 1:d]);
        @objective(model2, Min, 0);
        for k in 1:N
            @constraint(model2, Z[k] * (dot(D[k,:], x2) - B[k]) >= 0);
            # Objective
            if k % batch == 0
                iloc = Int(k / batch);
                optimize!(model2);
                AEs[iloc, :] = copy(value.(x2));
            end
        end
        model2 = nothing;

        # DP MLE
        DPMLEs = zeros((sizemax, d));
        model3 = JuMP.Model(Ipopt.Optimizer);
        set_silent(model3);
        @variable(model3, x3[u in 1:d]);
        for k in 1:N
            @constraint(model3, Z[k] * (dot(D[k,:], x3) - B[k]) >= 0);
            # Objective
            if k % batch == 0
                iloc = Int(k / batch);
                @objective(model3, Min, sum((x3 .- SOMLEs[iloc,:]).^2));
                optimize!(model3);
                DPMLEs[iloc, :] = copy(value.(x3));
            end
        end
        model3 = nothing;

        # LLE     
        LLEs = zeros((sizemax, d));
        for i in 1:sizemax
            Nloc = Int(i * batch);

            # Definition model
            model4 = JuMP.Model(Ipopt.Optimizer);
            set_silent(model4);
            @variable(model4, x4[u in 1:d]);

            # Objective
            @objective(model4, Min, sum((x4 .- SOMLEs[i,:]).^2) + 
                                    sum(log.(1 .+ exp.(Z[1:Nloc] .* (B[1:Nloc] .- sum(D[1:Nloc,:] .* x4', dims=2))))) / Nloc);
            
            # Solve
            optimize!(model4);
            LLEs[i,:] = copy(value.(x4));

            model4 = nothing;
        end


        # SP MLE     
        SPMLEs = zeros((sizemax, d));
        for i in 1:sizemax
            Nloc = Int(i * batch);

            # Definition model
            model5 = JuMP.Model(Ipopt.Optimizer);
            set_silent(model5);
            @variable(model5, x5[u in 1:d]);

            # Objective
            @objective(model5, Min, sum((x5 .- SOMLEs[i,:]).^2) + 
                                    sum(log.(1 .+ exp.(Zsto[1:Nloc] .* (B[1:Nloc] .- sum(D[1:Nloc,:] .* x5', dims=2))))) / Nloc);
            
            # Solve
            optimize!(model5);
            SPMLEs[i,:] = copy(value.(x5));

            model5 = nothing;
        end

        return μ, SOMLEs .- μ', WEs .- μ', AEs .- μ', DPMLEs .- μ', LLEs .- μ', SPMLEs .- μ';
	else
		@error "Not Implemented";
	end
end
    

# Run the experiments in parallel
@time data = pmap(
    (i,) -> runit(seed + i, dimension, Nsteps, instance, μmin, μmax, batch, sizemax),
    1:Nruns
);


# Save everything using JLD2.
@save "$(experiment_dir)$(experiment_name).dat" data Nruns dimension Nsteps instance μmin μmax batch sizemax;

