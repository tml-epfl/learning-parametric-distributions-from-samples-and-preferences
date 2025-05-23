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
experiments = filter(x -> isdir("$(exp_dir)$x") && occursin("_betterEstim_", x), readdir(exp_dir));
for experiment in experiments
    # Check for existing plots in the corresponding format
    existing_plots = length(filter(x -> occursin(".$format", x), readdir("$(exp_dir)$(experiment)/"))) != 0;

    if ! existing_plots
        println("Creating plots in .$format based on $(exp_dir)$(experiment) ...");

        name_experiment = split(experiment, ":")[2];
        @load "$(exp_dir)$(experiment)/$(name_experiment).dat" data Nruns Nsteps instance μmin μmax batch sizemax;

        μs = getindex.(data, 1);
        SOMLEs = permutedims(hcat(getindex.(data, 2)...));
        WEs = permutedims(hcat(getindex.(data, 3)...));
        RUs = permutedims(hcat(getindex.(data, 4)...));
        DPMLEs = permutedims(hcat(getindex.(data, 5)...));
        CEs = permutedims(hcat(getindex.(data, 6)...));
        TGEs = permutedims(hcat(getindex.(data, 7)...));
        TMLEs = permutedims(hcat(getindex.(data, 8)...));

        # Plot as function of n
        meanSOMLEs = mean(abs.(SOMLEs), dims=1)[1,:];
        stdSOMLEs = std(abs.(SOMLEs), dims=1)[1,:];
        meanWEs = mean(abs.(WEs), dims=1)[1,:];
        stdWEs = std(abs.(WEs), dims=1)[1,:];
        meanRUs = mean(abs.(RUs), dims=1)[1,:];
        stdRUs = std(abs.(RUs), dims=1)[1,:];
        meanDPMLEs = mean(abs.(DPMLEs), dims=1)[1,:];
        stdDPMLEs = std(abs.(DPMLEs), dims=1)[1,:];
        meanCEs = mean(abs.(CEs), dims=1)[1,:];
        stdCEs = std(abs.(CEs), dims=1)[1,:];
        meanTGEs = mean(abs.(TGEs), dims=1)[1,:];
        stdTGEs = std(abs.(TGEs), dims=1)[1,:];
        meanTMLEs = mean(abs.(TMLEs), dims=1)[1,:];
        stdTMLEs = std(abs.(TMLEs), dims=1)[1,:];

        cblind_colors = palette(:seaborn_colorblind6);
        dict_colors = Dict("SO" => cblind_colors[1],
                           "WE" => cblind_colors[2],
                           "RU" => cblind_colors[3],
                           "DP" => cblind_colors[4],
                           "CE" => cblind_colors[5],
                           "TGE" => cblind_colors[6]);

        start = batch;

        # Log-Log Plot Loss
        plot(start:Nsteps, meanSOMLEs[start:end], label="SO", linestyle=:auto, color=dict_colors["SO"], linewidth=2, legend=:bottomleft, yaxis=:log, xaxis=:log, xlabel=L"n", ylabel=L"|\widehat \theta_n - \theta^\star|", guidefontsize=18, tickfontsize=16, legendfontsize=12);
        plot!(start:Nsteps, meanWEs[start:end], label="WE", linestyle=:auto, color=dict_colors["WE"], linewidth=2);
        plot!(start:Nsteps, meanRUs[start:end], label="RU", linestyle=:auto, color=dict_colors["RU"], linewidth=2);
        plot!(start:Nsteps, meanDPMLEs[start:end], label="DP", linestyle=:auto, color=dict_colors["DP"], linewidth=2);
        plot!(start:Nsteps, meanCEs[start:end], label="CE", linestyle=:auto, color=dict_colors["CE"], linewidth=2);
        plot!([u*batch for u in 1:Int(sizemax)], meanTGEs, label="TrG", linestyle=:auto, color=dict_colors["TGE"], linewidth=2);
        plot!([u*batch for u in 1:Int(sizemax)], meanTMLEs, label="TrMLE", linestyle=:auto, color=:black, linewidth=2);
        file_plot = "$(exp_dir)$(experiment)/plot_betterEstim_inst_Gaussian_logplot_loss.$format";
        savefig(file_plot);
        
    else
        println("Plots in .$format based on $(exp_dir)$(experiment) already exist.");
    end
    println("");
end


