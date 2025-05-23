@everywhere using ArgParse, JLD2, Printf, JSON, Dates, IterTools, Random;
@everywhere using LinearAlgebra, Statistics, Distributions;
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
            default = "Matrices"
        "--instance"
            help = "Instance considered."
            arg_type = String
            default = "Gaussian"
        "--dimMax"
            help = "Dimensionality."
            arg_type = Int64
            default = 100
        "--Nruns"
            help = "Number of runs of the experiment."
            arg_type = Int64
            default = 1000000
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
dimMax = parsed_args["dimMax"];
Nruns = parsed_args["Nruns"];

# Naming files and folder
now_str = Dates.format(now(), "dd-mm_HHhMM");
experiment_name = "exp_" * expe * "inst_" * instance * "dMax_" * string(dimMax) * "_N" * string(Nruns);
experiment_dir = save_dir * now_str * ":" * experiment_name * "/";
mkdir(experiment_dir);
open("$(experiment_dir)parsed_args.json","w") do f
    JSON.print(f, parsed_args)
end

@everywhere σ(x) = 1/(1 + exp(-x));

@everywhere function runit(seed, N, instance, d)
	rng = MersenneTwister(seed);

	if instance == "Gaussian"
	    dist = Normal();

	    data = rand(rng, dist, (2, N, d));
	    X = d == 1 ? data[1, :] : data[1, :, :];
	    Y = d == 1 ? data[2, :] : data[1, :, :];

		XY = d == 1 ? X .* Y : sum(X .* Y, dims=2);	
		signXY = 2 * (XY .> 0 .- 1);
		absXY = abs.(XY);
		σXY = σ.(absXY);
		σmXY = σ.(-absXY);

		U1 = X .* σXY;
		U2 = X .* σmXY;
		U4 = X .* (2 .* σXY .- 1);
		U3 = Y .* signXY;

	    if d == 1	
			# ΔSP    
	    	ΔSP = U1 .* U2;
	    	meanΔSP = mean(ΔSP);
	    	stdΔSP = std(ΔSP);

			# ΔLLE
	    	ΔLLE = U4 .* U2;
		    meanΔLLE = mean(ΔLLE);
		    stdΔLLE = std(ΔLLE);

		    # RLLE
		    RLLE = U3 .* U2 .+ U2 .* U3;
		    meanRLLE = mean(RLLE);
		    stdRLLE = std(RLLE);
	    else	    
			# ΔSP    
		    meanΔSP = U1' * U2 / N;
		    stdΔSP = eigmin(meanΔSP);

			# ΔLLE
		    meanΔLLE = U4' * U2 / N;
		    stdΔLLE = eigmin(meanΔLLE);

		    # RLLE
		    meanRLLE = (U3' * U2 .+ U2' * U3) / N;
		    stdRLLE = eigmin(meanRLLE);
	    end
	else
		@error "Not Implemented";
	end
	return meanΔSP, stdΔSP, meanΔLLE, stdΔLLE, meanRLLE, stdRLLE;
end
    
@time data = pmap(
    (d,) -> runit(seed, Nruns, instance, d),
    1:dimMax
);

# Save everything using JLD2.
@save "$(experiment_dir)$(experiment_name).dat" data Nruns instance dimMax;


