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
experiments = filter(x -> isdir("$(exp_dir)$x") && occursin("_errorlarged_", x), readdir(exp_dir));
for experiment in experiments
    # Check for existing plots in the corresponding format
    existing_plots = length(filter(x -> occursin(".$format", x), readdir("$(exp_dir)$(experiment)/"))) != 0;

    if ! existing_plots
        println("Creating plots in .$format based on $(exp_dir)$(experiment) ...");

        name_experiment = split(experiment, ":")[2];
        @load "$(exp_dir)$(experiment)/$(name_experiment).dat" data Nruns dimension Nsteps instance μmin μmax batch sizemax

        μs = getindex.(data, 1);
        SOMLEs = permutedims(cat(getindex.(data, 2)...; dims=3), (3, 1, 2));
        WEs = permutedims(cat(getindex.(data, 3)...; dims=3), (3, 1, 2));
        AEs = permutedims(cat(getindex.(data, 4)...; dims=3), (3, 1, 2));
        DPMLEs = permutedims(cat(getindex.(data, 5)...; dims=3), (3, 1, 2));
        LLEs = permutedims(cat(getindex.(data, 6)...; dims=3), (3, 1, 2));
        SPMLEs = permutedims(cat(getindex.(data, 7)...; dims=3), (3, 1, 2));

        # Plot as function of n
        normSOMLEs = sqrt.(sum(SOMLEs.^2,dims=3)[:,:]);
        meanSOMLEs = mean(normSOMLEs, dims=1)[1,:];
        stdSOMLEs = std(normSOMLEs, dims=1)[1,:];
        inv_meanSOMLEs = 1 ./ meanSOMLEs;

        normWEs = sqrt.(sum(WEs.^2,dims=3)[:,:]);
        meanWEs = mean(normWEs, dims=1)[1,:];
        stdWEs = std(normWEs, dims=1)[1,:];
        inv_meanWEs = 1 ./ meanWEs;

        normAEs = sqrt.(sum(AEs.^2,dims=3)[:,:]);
        meanAEs = mean(normAEs, dims=1)[1,:];
        stdAEs = std(normAEs, dims=1)[1,:];
        inv_meanAEs = 1 ./ meanAEs;

        normDPMLEs = sqrt.(sum(DPMLEs.^2,dims=3)[:,:]);
        meanDPMLEs = mean(normDPMLEs, dims=1)[1,:];
        stdDPMLEs = std(normDPMLEs, dims=1)[1,:];
        inv_meanDPMLEs = 1 ./ meanDPMLEs;

        normLLEs = sqrt.(sum(LLEs.^2,dims=3)[:,:]);
        meanLLEs = mean(normLLEs, dims=1)[1,:];
        stdLLEs = std(normLLEs, dims=1)[1,:];
        inv_meanLLEs = 1 ./ meanLLEs;

        normSPMLEs = sqrt.(sum(SPMLEs.^2,dims=3)[:,:]);
        meanSPMLEs = mean(normSPMLEs, dims=1)[1,:];
        stdSPMLEs = std(normSPMLEs, dims=1)[1,:];
        inv_meanSPMLEs = 1 ./ meanSPMLEs;

        cblind_colors = palette(:seaborn_colorblind6);
        dict_colors = Dict("SO" => cblind_colors[1],
                           "WE" => cblind_colors[2],
                           "AE" => cblind_colors[3],
                           "DP" => cblind_colors[4],
                           "LLE" => cblind_colors[5],
                           "SP" => cblind_colors[6]);

        start = batch;

        # Plot Loss
        plot([u*batch for u in 1:Int(sizemax)], meanSOMLEs, ribbon=stdSOMLEs, fillalpha=.3, label="SO", linestyle=:auto, color=dict_colors["SO"], linewidth=2, legend=:topright, xlabel=L"n", ylabel=L"||\widehat \theta_n - \theta^\star||_{2}", guidefontsize=18, tickfontsize=16, legendfontsize=12);
        plot!([u*batch for u in 1:Int(sizemax)], meanWEs, ribbon=stdWEs, fillalpha=.3, label="WE", linestyle=:auto, color=dict_colors["WE"], linewidth=2);
        plot!([u*batch for u in 1:Int(sizemax)], meanAEs, ribbon=stdAEs, fillalpha=.3, label="AE", linestyle=:auto, color=dict_colors["AE"], linewidth=2);
        plot!([u*batch for u in 1:Int(sizemax)], meanDPMLEs, ribbon=stdDPMLEs, fillalpha=.3, label="DP", linestyle=:auto, color=dict_colors["DP"], linewidth=2);
        plot!([u*batch for u in 1:Int(sizemax)], meanSPMLEs, ribbon=stdSPMLEs, fillalpha=.3, label="SP (sto)", linestyle=:auto, color=dict_colors["SP"], linewidth=2);
        plot!([u*batch for u in 1:Int(sizemax)], meanLLEs, ribbon=stdLLEs, fillalpha=.3, label="SP (det)", linestyle=:auto, color=dict_colors["LLE"], linewidth=2);
        file_plot = "$(exp_dir)$(experiment)/plot_errord_" * string(dimension) * "_inst_Gaussian_loss.$format";
        savefig(file_plot);

        # Plot Inv loss
        plot([u*batch for u in 1:Int(sizemax)], inv_meanWEs, fillalpha=.3, label="WE", linestyle=:auto, color=dict_colors["WE"], linewidth=2, legend=:topleft, xlabel=L"n", ylabel=L"1/||\widehat \theta_n - \theta^\star||_{2}", guidefontsize=18, tickfontsize=16, legendfontsize=12);
        plot!([u*batch for u in 1:Int(sizemax)], inv_meanAEs, fillalpha=.3, label="AE", linestyle=:auto, color=dict_colors["AE"], linewidth=2);
        plot!([u*batch for u in 1:Int(sizemax)], inv_meanDPMLEs, fillalpha=.3, label="DP", linestyle=:auto, color=dict_colors["DP"], linewidth=2);
        file_plot = "$(exp_dir)$(experiment)/plot_errord_" * string(dimension) * "_inst_Gaussian_invloss.$format";
        savefig(file_plot);

        # Log-Log Plot Loss
        plot([u*batch for u in 1:Int(sizemax)], meanSOMLEs, label="SO", linestyle=:auto, color=dict_colors["SO"], linewidth=2, legend=:bottomleft, yaxis=:log, xaxis=:log, xlabel=L"n", ylabel=L"||\widehat \theta_n - \theta^\star||_{2}", guidefontsize=18, tickfontsize=16, legendfontsize=12);
        plot!([u*batch for u in 1:Int(sizemax)], meanWEs, label="WE", linestyle=:auto, color=dict_colors["WE"], linewidth=2);
        plot!([u*batch for u in 1:Int(sizemax)], meanAEs, label="AE", linestyle=:auto, color=dict_colors["AE"], linewidth=2);
        plot!([u*batch for u in 1:Int(sizemax)], meanDPMLEs, label="DP", linestyle=:auto, color=dict_colors["DP"], linewidth=2);
        plot!([u*batch for u in 1:Int(sizemax)], meanSPMLEs, label="SP (sto)", linestyle=:auto, color=dict_colors["SP"], linewidth=2);
        plot!([u*batch for u in 1:Int(sizemax)], meanLLEs, label="SP (det)", linestyle=:auto, color=dict_colors["LLE"], linewidth=2);
        file_plot = "$(exp_dir)$(experiment)/plot_errord_" * string(dimension) * "_inst_Gaussian_logplot_loss.$format";
        savefig(file_plot);
        
    else
        println("Plots in .$format based on $(exp_dir)$(experiment) already exist.");
    end
    println("");
end


