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
experiments = filter(x -> isdir("$(exp_dir)$x") && occursin("_Matrices", x) && occursin("dMax_", x), readdir(exp_dir));
for experiment in experiments
    # Check for existing plots in the corresponding format
    existing_plots = length(filter(x -> occursin(".$format", x), readdir("$(exp_dir)$(experiment)/"))) != 0;

    if ! existing_plots
        println("Creating plots in .$format based on $(exp_dir)$(experiment) ...");

        name_experiment = split(experiment, ":")[2];
        @load "$(exp_dir)$(experiment)/$(name_experiment).dat" data Nruns instance dimMax;

        rangeDims = collect(1:dimMax);
        sizeD = size(rangeDims)[1];

        meanΔSP = getindex.(data, 1);
        stdΔSP = getindex.(data, 2);
        meanΔLLE = getindex.(data, 3);
        stdΔLLE = getindex.(data, 4);
        meanRLLE = getindex.(data, 5);
        stdRLLE = getindex.(data, 6);

        cblind_colors = palette(:seaborn_colorblind6);
        dict_colors = Dict("SO" => cblind_colors[1],
                           "WE" => cblind_colors[2],
                           "AE" => cblind_colors[3],
                           "DP" => cblind_colors[4],
                           "LLE" => cblind_colors[5],
                           "SP" => cblind_colors[6]);

        errorΔSP = zeros(dimMax);
        scaleΔSP = zeros(dimMax);
        errorΔLLE = zeros(dimMax);
        scaleΔLLE = zeros(dimMax);
        errorRLLE = zeros(dimMax);
        scaleRLLE = zeros(dimMax);
        for i in 1:dimMax
            if i == 1        
                scaleΔSP[i] = meanΔSP[i];
                errorΔSP[i] = 0.;
                scaleΔLLE[i] = meanΔLLE[i];
                errorΔLLE[i] = 0.;
                scaleRLLE[i] = meanRLLE[i];
                errorRLLE[i] = 0.;
            else
                scaleΔSP[i] = mean(meanΔSP[i]);
                errorΔSP[i] = norm(meanΔSP[i] .- scaleΔSP[i] * Diagonal(ones(i)));
                scaleΔLLE[i] = mean(meanΔLLE[i]);
                errorΔLLE[i] = norm(meanΔLLE[i] .- scaleΔLLE[i] * Diagonal(ones(i)));
                scaleRLLE[i] = mean(meanRLLE[i]);
                errorRLLE[i] = norm(meanRLLE[i] .- scaleRLLE[i] * Diagonal(ones(i)));
            end
        end

        # Plot of scaling 
        plot(1:dimMax, scaleΔSP, label="Δ SP (sto)", linestyle=:auto, color=dict_colors["SO"], yaxis=:log, linewidth=2, legend=:topright, xlabel=L"d", ylabel=L"\alpha_d", guidefontsize=18, tickfontsize=16, legendfontsize=12);
        plot!(1:dimMax, scaleΔLLE, label="Δ SP (det)", linestyle=:auto, color=dict_colors["WE"], linewidth=2);
        plot!(1:dimMax, scaleRLLE, fillalpha=.3, label="R SP (det)", linestyle=:auto, color=dict_colors["AE"], linewidth=2);
        file_plot = "$(exp_dir)$(experiment)/plot_scalingMatrices.$format";
        savefig(file_plot);

        # Plot of error 
        plot(1:dimMax, errorΔSP, label="Δ SP (sto)", linestyle=:auto, color=dict_colors["SO"], linewidth=2, legend=:topright, xlabel=L"d", ylabel=L"|| Δ - \alpha_d I_d ||_2", guidefontsize=18, tickfontsize=16, legendfontsize=12);
        plot!(1:dimMax, errorΔLLE, label="Δ SP (det)", linestyle=:auto, color=dict_colors["WE"], linewidth=2);
        plot!(1:dimMax, errorRLLE, fillalpha=.3, label="R SP (det)", linestyle=:auto, color=dict_colors["AE"], linewidth=2);
        file_plot = "$(exp_dir)$(experiment)/plot_errorMatrices.$format";
        savefig(file_plot);
        
    else
        println("Plots in .$format based on $(exp_dir)$(experiment) already exist.");
    end
    println("");
end


