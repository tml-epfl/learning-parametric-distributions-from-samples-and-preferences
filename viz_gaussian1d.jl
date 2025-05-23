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
experiments = filter(x -> isdir("$(exp_dir)$x") && occursin("_error1d_", x), readdir(exp_dir));
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
        LLEs = permutedims(hcat(getindex.(data, 6)...));
        SPMLEs = permutedims(hcat(getindex.(data, 7)...));

        # Plot as function of n
        meanSOMLEs = mean(abs.(SOMLEs), dims=1)[1,:];
        stdSOMLEs = std(abs.(SOMLEs), dims=1)[1,:];
        meanWEs = mean(abs.(WEs), dims=1)[1,:];
        stdWEs = std(abs.(WEs), dims=1)[1,:];
        inv_meanWEs = 1 ./ meanWEs;
        meanRUs = mean(abs.(RUs), dims=1)[1,:];
        stdRUs = std(abs.(RUs), dims=1)[1,:];
        inv_meanRUs = 1 ./ meanRUs;
        meanDPMLEs = mean(abs.(DPMLEs), dims=1)[1,:];
        stdDPMLEs = std(abs.(DPMLEs), dims=1)[1,:];
        inv_meanDPMLEs = 1 ./ meanDPMLEs;
        meanLLEs = mean(abs.(LLEs), dims=1)[1,:];
        stdLLEs = std(abs.(LLEs), dims=1)[1,:];
        meanSPMLEs = mean(abs.(SPMLEs), dims=1)[1,:];
        stdSPMLEs = std(abs.(SPMLEs), dims=1)[1,:];

        # Plot limiting Gaussian distribution
        lastSOMLE = SOMLEs[:,end];
        lastSPMLE = SPMLEs[:,end];
        lastLLE = SPMLEs[:,end];

        cblind_colors = palette(:seaborn_colorblind6);
        dict_colors = Dict("SO" => cblind_colors[1],
                           "WE" => cblind_colors[2],
                           "RU" => cblind_colors[3],
                           "DP" => cblind_colors[4],
                           "LLE" => cblind_colors[5],
                           "SP" => cblind_colors[6]);

        start = batch;

        # Plot Loss
        plot(start:Nsteps, meanSOMLEs[start:end], ribbon=stdSOMLEs[start:end], fillalpha=.3, label="SO", linestyle=:auto, color=dict_colors["SO"], linewidth=2, legend=:topright, xlabel=L"n", ylabel=L"|\widehat \theta_n - \theta^\star|", guidefontsize=18, tickfontsize=16, legendfontsize=12);
        plot!(start:Nsteps, meanWEs[start:end], ribbon=stdWEs[start:end], fillalpha=.3, label="WE", linestyle=:auto, color=dict_colors["WE"], linewidth=2);
        plot!(start:Nsteps, meanRUs[start:end], ribbon=stdRUs[start:end], fillalpha=.3, label="RU", linestyle=:auto, color=dict_colors["RU"], linewidth=2);
        plot!(start:Nsteps, meanDPMLEs[start:end], ribbon=stdDPMLEs[start:end], fillalpha=.3, label="DP", linestyle=:auto, color=dict_colors["DP"], linewidth=2);
        plot!([u*batch for u in 1:Int(sizemax)], meanSPMLEs, ribbon=stdSPMLEs, fillalpha=.3, label="SP (sto)", linestyle=:auto, color=dict_colors["SP"], linewidth=2);
        plot!([u*batch for u in 1:Int(sizemax)], meanLLEs, ribbon=stdLLEs, fillalpha=.3, label="SP (det)", linestyle=:auto, color=dict_colors["LLE"], linewidth=2);
        file_plot = "$(exp_dir)$(experiment)/plot_error1d_inst_Gaussian_loss.$format";
        savefig(file_plot);

        # Plot Inv loss
        plot(start:Nsteps, inv_meanWEs[start:end], fillalpha=.3, label="WE", linestyle=:auto, color=dict_colors["WE"], linewidth=2, legend=:topleft, xlabel=L"n", ylabel=L"1/|\widehat \theta_n - \theta^\star|", guidefontsize=18, tickfontsize=16, legendfontsize=12);
        plot!(start:Nsteps, inv_meanRUs[start:end], fillalpha=.3, label="RU", linestyle=:auto, color=dict_colors["RU"], linewidth=2);
        plot!(start:Nsteps, inv_meanDPMLEs[start:end], fillalpha=.3, label="DP", linestyle=:auto, color=dict_colors["DP"], linewidth=2);
        file_plot = "$(exp_dir)$(experiment)/plot_error1d_inst_Gaussian_invloss.$format";
        savefig(file_plot);

        # Log-Log Plot Loss
        plot(start:Nsteps, meanSOMLEs[start:end], label="SO", linestyle=:auto, color=dict_colors["SO"], linewidth=2, legend=:bottomleft, yaxis=:log, xaxis=:log, xlabel=L"n", ylabel=L"|\widehat \theta_n - \theta^\star|", guidefontsize=18, tickfontsize=16, legendfontsize=12);
        plot!(start:Nsteps, meanWEs[start:end], label="WE", linestyle=:auto, color=dict_colors["WE"], linewidth=2);
        plot!(start:Nsteps, meanRUs[start:end], label="RU", linestyle=:auto, color=dict_colors["RU"], linewidth=2);
        plot!(start:Nsteps, meanDPMLEs[start:end], label="DP", linestyle=:auto, color=dict_colors["DP"], linewidth=2);
        plot!([u*batch for u in 1:Int(sizemax)], meanSPMLEs, label="SP (sto)", linestyle=:auto, color=dict_colors["SP"], linewidth=2);
        plot!([u*batch for u in 1:Int(sizemax)], meanLLEs, label="SP (det)", linestyle=:auto, color=dict_colors["LLE"], linewidth=2);
        file_plot = "$(exp_dir)$(experiment)/plot_error1d_inst_Gaussian_logplot_loss.$format";
        savefig(file_plot);
        
    else
        println("Plots in .$format based on $(exp_dir)$(experiment) already exist.");
    end
    println("");
end


