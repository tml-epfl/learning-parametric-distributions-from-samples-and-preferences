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
experiments = filter(x -> isdir("$(exp_dir)$x") && occursin("_normregu_", x), readdir(exp_dir));
for experiment in experiments
    # Check for existing plots in the corresponding format
    existing_plots = length(filter(x -> occursin(".$format", x), readdir("$(exp_dir)$(experiment)/"))) != 0;

    if ! existing_plots
        println("Creating plots in .$format based on $(exp_dir)$(experiment) ...");

        name_experiment = split(experiment, ":")[2];
        @load "$(exp_dir)$(experiment)/$(name_experiment).dat" data Nruns Nsteps instance μmin μmax batch sizemax βs λs;

        μs = getindex.(data, 1);
        SOMLEs = permutedims(hcat(getindex.(data, 2)...));
        DPMLEs = permutedims(hcat(getindex.(data, 3)...));
        SPEλs = permutedims(cat(getindex.(data, 4)...; dims=3), (1, 3, 2));
        SPEβs = permutedims(cat(getindex.(data, 5)...; dims=3), (1, 3, 2));

        # Plot as function of n
        meanSOMLEs = mean(abs.(SOMLEs), dims=1)[1,:];
        stdSOMLEs = std(abs.(SOMLEs), dims=1)[1,:];
        meanDPMLEs = mean(abs.(DPMLEs), dims=1)[1,:];
        stdDPMLEs = std(abs.(DPMLEs), dims=1)[1,:];
        meanSPEλs = mean(abs.(SPEλs), dims=2)[:,1,:];
        stdSPEλs = std(abs.(SPEλs), dims=2)[:,1,:];
        meanSPEβs = mean(abs.(SPEβs), dims=2)[:,1,:];
        stdSPEβs = std(abs.(SPEβs), dims=2)[:,1,:];

        paired_colors = palette(:Paired_12);

        start = batch;

        # Log-Log Plot Loss
        plot(start:Nsteps, meanSOMLEs[start:end], label="SO", linestyle=:auto, color=paired_colors[11], linewidth=2, legend=:bottomleft, yaxis=:log, xaxis=:log, xlabel=L"n", ylabel=L"|\widehat \theta_n - \theta^\star|", guidefontsize=18, tickfontsize=16, legendfontsize=12);
        plot!(start:Nsteps, meanDPMLEs[start:end], label="DP", linestyle=:auto, color=paired_colors[12], linewidth=2);
        for (k, λk) in enumerate(λs)
            plot!([u*batch for u in 1:Int(sizemax)], meanSPEλs[k,:], label=L"λ = " * string(λk), linestyle=:auto, color=paired_colors[Int(2*k-1)], linewidth=2);
        end
        for (k, βk) in enumerate(βs)
            plot!([u*batch for u in 1:Int(sizemax)], meanSPEβs[k,:], label=L"β = " * string(βk), linestyle=:auto, color=paired_colors[Int(2*k)], linewidth=2);
        end  
        file_plot = "$(exp_dir)$(experiment)/plot_normregu_inst_Gaussian_logplot_loss.$format";
        savefig(file_plot);

        # Normalization only
        plot(start:Nsteps, meanSOMLEs[start:end], label="SO", linestyle=:auto, color=paired_colors[11], linewidth=2, legend=:bottomleft, yaxis=:log, xaxis=:log, xlabel=L"n", ylabel=L"|\widehat \theta_n - \theta^\star|", guidefontsize=18, tickfontsize=16, legendfontsize=12);
        plot!(start:Nsteps, meanDPMLEs[start:end], label="DP", linestyle=:auto, color=paired_colors[12], linewidth=2);
        for (k, βk) in enumerate(βs)
            if βk > 0.08 && βk < 20
                plot!([u*batch for u in 1:Int(sizemax)], meanSPEβs[k,:], label=L"β = " * string(βk), linestyle=:auto, color=paired_colors[Int(2*k)], linewidth=2);
            end
        end  
        file_plot = "$(exp_dir)$(experiment)/plot_norm_inst_Gaussian_logplot_loss.$format";
        savefig(file_plot);

        # Regularization only
        plot(start:Nsteps, meanSOMLEs[start:end], label="SO", linestyle=:auto, color=paired_colors[11], linewidth=2, legend=:bottomleft, yaxis=:log, xaxis=:log, xlabel=L"n", ylabel=L"|\widehat \theta_n - \theta^\star|", guidefontsize=18, tickfontsize=16, legendfontsize=12);
        plot!(start:Nsteps, meanDPMLEs[start:end], label="DP", linestyle=:auto, color=paired_colors[12], linewidth=2);
        for (k, λk) in enumerate(λs)
            if λk > 0.08 && λk < 20
                plot!([u*batch for u in 1:Int(sizemax)], meanSPEλs[k,:], label=L"λ = " * string(λk), linestyle=:auto, color=paired_colors[Int(2*k-1)], linewidth=2);
            end
        end
        file_plot = "$(exp_dir)$(experiment)/plot_regu_inst_Gaussian_logplot_loss.$format";
        savefig(file_plot);
        
    else
        println("Plots in .$format based on $(exp_dir)$(experiment) already exist.");
    end
    println("");
end


