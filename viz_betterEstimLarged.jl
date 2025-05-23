using Random, JLD2, ArgParse;
using LinearAlgebra, Statistics, StatsBase, Plots;
using StatsPlots, ColorSchemes;
using LaTeXStrings;

function parse_commandline()
    s = ArgParseSettings();

    @add_arg_table! s begin
        "--exp_dir"
            help = "Directory for loading the experiment data."
            arg_type = String
            default = "experiments/"
        "--format"
            help = "Format to save the figures."
            arg_type = String
            default = "pdf"
    end

    parse_args(s);
end

# Parameters
parsed_args = parse_commandline();
exp_dir = parsed_args["exp_dir"];
format = parsed_args["format"];

# Loading
experiments = filter(x -> isdir("$(exp_dir)$x") && occursin("_betterEstimlarged_", x), readdir(exp_dir));
for experiment in experiments
    # Check for existing plots in the corresponding format
    existing_plots = length(filter(x -> occursin(".$format", x), readdir("$(exp_dir)$(experiment)/"))) != 0;

    if ! existing_plots
        println("Creating plots in .$format based on $(exp_dir)$(experiment) ...");

        name_experiment = split(experiment, ":")[2];
        @load "$(exp_dir)$(experiment)/$(name_experiment).dat" data Nruns dimension Nsteps instance μmin μmax batch sizemax

        μs = getindex.(data, 1);
        SOMLEs = permutedims(cat(getindex.(data, 2)...; dims=3), (3, 1, 2));
        AEs = permutedims(cat(getindex.(data, 3)...; dims=3), (3, 1, 2));
        DPMLEs = permutedims(cat(getindex.(data, 4)...; dims=3), (3, 1, 2));
        CCEs = permutedims(cat(getindex.(data, 5)...; dims=3), (3, 1, 2));

        # Plot as function of n
        normSOMLEs = sqrt.(sum(SOMLEs.^2,dims=3)[:,:]);
        meanSOMLEs = mean(normSOMLEs, dims=1)[1,:];
        stdSOMLEs = std(normSOMLEs, dims=1)[1,:];

        normAEs = sqrt.(sum(AEs.^2,dims=3)[:,:]);
        meanAEs = mean(normAEs, dims=1)[1,:];
        stdAEs = std(normAEs, dims=1)[1,:];

        normDPMLEs = sqrt.(sum(DPMLEs.^2,dims=3)[:,:]);
        meanDPMLEs = mean(normDPMLEs, dims=1)[1,:];
        stdDPMLEs = std(normDPMLEs, dims=1)[1,:];

        normCCEs = sqrt.(sum(CCEs.^2,dims=3)[:,:]);
        meanCCEs = mean(normCCEs, dims=1)[1,:];
        stdCCEs = std(normCCEs, dims=1)[1,:];

        cblind_colors = palette(:seaborn_colorblind6);
        dict_colors = Dict("SO" => cblind_colors[1],
                           "AE" => cblind_colors[3],
                           "DP" => cblind_colors[4],
                           "CCE" => cblind_colors[2]);

        start = batch;

        # Log-Log Plot Loss
        plot([u*batch for u in 1:Int(sizemax)], meanSOMLEs, label="SO", linestyle=:auto, color=dict_colors["SO"], linewidth=2, legend=:bottomleft, yaxis=:log, xaxis=:log, xlabel=L"n", ylabel=L"||\widehat \theta_n - \theta^\star||_{2}", guidefontsize=18, tickfontsize=16, legendfontsize=12);
        plot!([u*batch for u in 1:Int(sizemax)], meanAEs, label="AE", linestyle=:auto, color=dict_colors["AE"], linewidth=2);
        plot!([u*batch for u in 1:Int(sizemax)], meanDPMLEs, label="DP", linestyle=:auto, color=dict_colors["DP"], linewidth=2);
        plot!([u*batch for u in 1:Int(sizemax)], meanCCEs, label="CCE", linestyle=:auto, color=dict_colors["CCE"], linewidth=2);
        file_plot = "$(exp_dir)$(experiment)/plot_betterEstimd_" * string(dimension) * "_inst_Gaussian_logplot_loss.$format";
        savefig(file_plot);
        
    else
        println("Plots in .$format based on $(exp_dir)$(experiment) already exist.");
    end
    println("");
end


