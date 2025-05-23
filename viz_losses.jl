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
experiments = filter(x -> isdir("$(exp_dir)$x") && occursin("_losses_", x), readdir(exp_dir));
for experiment in experiments
    # Check for existing plots in the corresponding format
    existing_plots = length(filter(x -> occursin(".$format", x), readdir("$(exp_dir)$(experiment)/"))) != 0;

    if ! existing_plots
        println("Creating plots in .$format based on $(exp_dir)$(experiment) ...");

        name_experiment = split(experiment, ":")[2];
        @load "$(exp_dir)$(experiment)/$(name_experiment).dat" data Nruns Nsteps instance μmin μmax batch sizemax;

        μs = getindex.(data, 1);
        SOMLEs = permutedims(hcat(getindex.(data, 2)...));
        DPMLEs = permutedims(hcat(getindex.(data, 3)...));
        SPEs = permutedims(hcat(getindex.(data, 4)...));
        HingEs = permutedims(hcat(getindex.(data, 5)...));
        SquarEs = permutedims(hcat(getindex.(data, 6)...));
        TruncSquarEs = permutedims(hcat(getindex.(data, 7)...));
        SavagEs = permutedims(hcat(getindex.(data, 8)...));
        ExpEs = permutedims(hcat(getindex.(data, 9)...));

        # Plot as function of n
        meanSOMLEs = mean(abs.(SOMLEs), dims=1)[1,:];
        stdSOMLEs = std(abs.(SOMLEs), dims=1)[1,:];
        meanDPMLEs = mean(abs.(DPMLEs), dims=1)[1,:];
        stdDPMLEs = std(abs.(DPMLEs), dims=1)[1,:];
        meanSPEs = mean(abs.(SPEs), dims=1)[1,:];
        stdSPEs = std(abs.(SPEs), dims=1)[1,:];
        meanHingEs = mean(abs.(HingEs), dims=1)[1,:];
        stdHingEs = std(abs.(HingEs), dims=1)[1,:];
        meanSquarEs = mean(abs.(SquarEs), dims=1)[1,:];
        stdSquarEs = std(abs.(SquarEs), dims=1)[1,:];
        meanTruncSquarEs = mean(abs.(TruncSquarEs), dims=1)[1,:];
        stdTruncSquarEs = std(abs.(TruncSquarEs), dims=1)[1,:];
        meanSavagEs = mean(abs.(SavagEs), dims=1)[1,:];
        stdSavagEs = std(abs.(SavagEs), dims=1)[1,:];
        meanExpEs = mean(abs.(ExpEs), dims=1)[1,:];
        stdExpEs = std(abs.(ExpEs), dims=1)[1,:];

        paired_colors = palette(:Paired_12);

        start = batch;

        # Log-Log Plot Loss
        plot(start:Nsteps, meanSOMLEs[start:end], label="SO", linestyle=:auto, color=paired_colors[11], linewidth=2, legend=:bottomleft, yaxis=:log, xaxis=:log, xlabel=L"n", ylabel=L"|\widehat \theta_n - \theta^\star|", guidefontsize=18, tickfontsize=16, legendfontsize=12);
        plot!(start:Nsteps, meanDPMLEs[start:end], label="DP", linestyle=:auto, color=paired_colors[12], linewidth=2);
        plot!([u*batch for u in 1:Int(sizemax)], meanSPEs, label="Log", linestyle=:auto, color=paired_colors[1], linewidth=2);
        plot!([u*batch for u in 1:Int(sizemax)], meanHingEs, label="Hin", linestyle=:auto, color=paired_colors[3], linewidth=2);
        plot!([u*batch for u in 1:Int(sizemax)], meanSquarEs, label="Squ", linestyle=:auto, color=paired_colors[5], linewidth=2);
        plot!([u*batch for u in 1:Int(sizemax)], meanTruncSquarEs, label="TrS", linestyle=:auto, color=paired_colors[6], linewidth=2);
        plot!([u*batch for u in 1:Int(sizemax)], meanSavagEs, label="Sav", linestyle=:auto, color=paired_colors[7], linewidth=2);
        plot!([u*batch for u in 1:Int(sizemax)], meanExpEs, label="Exp", linestyle=:auto, color=paired_colors[9], linewidth=2);
        file_plot = "$(exp_dir)$(experiment)/plot_losses_inst_Gaussian_logplot_loss.$format";
        savefig(file_plot);
        
    else
        println("Plots in .$format based on $(exp_dir)$(experiment) already exist.");
    end
    println("");
end


