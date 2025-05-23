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
experiments = filter(x -> isdir("$(exp_dir)$x") && occursin("_errorvard_", x), readdir(exp_dir));
for experiment in experiments
    # Check for existing plots in the corresponding format
    existing_plots = length(filter(x -> occursin(".$format", x), readdir("$(exp_dir)$(experiment)/"))) != 0;

    if ! existing_plots
        println("Creating plots in .$format based on $(exp_dir)$(experiment) ...");

        name_experiment = split(experiment, ":")[2];
        @load "$(exp_dir)$(experiment)/$(name_experiment).dat" data Nruns dimMax dimStep Nsteps instance μmin μmax;

        rangeDims = collect(1:dimStep:dimMax) .+ 1;
        sizeD = size(rangeDims)[1];

        μs = getindex.(data, 1);
        SOMLEs = getindex.(data, 2);
        WEs = getindex.(data, 3);
        AEs = getindex.(data, 4);
        DPMLEs = getindex.(data, 5);
        LLEs = getindex.(data, 6);
        SPMLEs = getindex.(data, 7);

        meanSOMLEs = zeros(sizeD);
        stdSOMLEs = zeros(sizeD);
        meanWEs = zeros(sizeD);
        stdWEs = zeros(sizeD);
        meanAEs = zeros(sizeD);
        stdAEs = zeros(sizeD);
        meanDPMLEs = zeros(sizeD);
        stdDPMLEs = zeros(sizeD);
        meanLLEs = zeros(sizeD);
        stdLLEs = zeros(sizeD);
        meanSPMLEs = zeros(sizeD);
        stdSPMLEs = zeros(sizeD);
        for i in 1:sizeD
            valSOMLE = permutedims(cat(SOMLEs[i,:]..., dims=2), (2,1));
            normSOMLEs = sqrt.(sum(valSOMLE.^2,dims=2)[:,1]);
            meanSOMLEs[i] = mean(normSOMLEs);
            stdSOMLEs[i] = std(normSOMLEs);

            valWE = permutedims(cat(WEs[i,:]..., dims=2), (2,1));
            normWEs = sqrt.(sum(valWE.^2,dims=2)[:,1]);
            meanWEs[i] = mean(normWEs);
            stdWEs[i] = std(normWEs);

            valAE = permutedims(cat(AEs[i,:]..., dims=2), (2,1));
            normAEs = sqrt.(sum(valAE.^2,dims=2)[:,1]);
            meanAEs[i] = mean(normAEs);
            stdAEs[i] = std(normAEs);

            valDPMLE = permutedims(cat(DPMLEs[i,:]..., dims=2), (2,1));
            normDPMLEs = sqrt.(sum(valDPMLE.^2,dims=2)[:,1]);
            meanDPMLEs[i] = mean(normDPMLEs);
            stdDPMLEs[i] = std(normDPMLEs);

            valLLE = permutedims(cat(LLEs[i,:]..., dims=2), (2,1));
            normLLEs = sqrt.(sum(valLLE.^2,dims=2)[:,1]);
            meanLLEs[i] = mean(normLLEs);
            stdLLEs[i] = std(normLLEs);

            valSPMLE = permutedims(cat(SPMLEs[i,:]..., dims=2), (2,1));
            normSPMLEs = sqrt.(sum(valSPMLE.^2,dims=2)[:,1]);
            meanSPMLEs[i] = mean(normSPMLEs);
            stdSPMLEs[i] = std(normSPMLEs);
        end
        inv_meanSOMLEs = 1 ./ meanSOMLEs;
        inv_meanWEs = 1 ./ meanWEs;
        inv_meanAEs = 1 ./ meanAEs;
        inv_meanDPMLEs = 1 ./ meanDPMLEs;
        inv_meanLLEs = 1 ./ meanLLEs;
        inv_meanSPMLEs = 1 ./ meanSPMLEs;

        cblind_colors = palette(:seaborn_colorblind6);
        dict_colors = Dict("SO" => cblind_colors[1],
                           "WE" => cblind_colors[2],
                           "AE" => cblind_colors[3],
                           "DP" => cblind_colors[4],
                           "LLE" => cblind_colors[5],
                           "SP" => cblind_colors[6]);

        # Plot Loss
        plot(rangeDims, meanSOMLEs, ribbon=stdSOMLEs, fillalpha=.3, label="SO", linestyle=:auto, color=dict_colors["SO"], linewidth=2, legend=:topleft, xlabel=L"d", ylabel=L"||\widehat \theta_n - \theta^\star||_{2}", guidefontsize=18, tickfontsize=16, legendfontsize=12);
        plot!(rangeDims, meanWEs, ribbon=stdWEs, fillalpha=.3, label="WE", linestyle=:auto, color=dict_colors["WE"], linewidth=2);
        plot!(rangeDims, meanAEs, ribbon=stdAEs, fillalpha=.3, label="AE", linestyle=:auto, color=dict_colors["AE"], linewidth=2);
        plot!(rangeDims, meanDPMLEs, ribbon=stdDPMLEs, fillalpha=.3, label="DP", linestyle=:auto, color=dict_colors["DP"], linewidth=2);
        plot!(rangeDims, meanSPMLEs, ribbon=stdSPMLEs, fillalpha=.3, label="SP (sto)", linestyle=:auto, color=dict_colors["SP"], linewidth=2);
        plot!(rangeDims, meanLLEs, ribbon=stdLLEs, fillalpha=.3, label="SP (det)", linestyle=:auto, color=dict_colors["LLE"], linewidth=2);
        file_plot = "$(exp_dir)$(experiment)/plot_errorvard_inst_Gaussian_loss.$format";
        savefig(file_plot);
        
    else
        println("Plots in .$format based on $(exp_dir)$(experiment) already exist.");
    end
    println("");
end


