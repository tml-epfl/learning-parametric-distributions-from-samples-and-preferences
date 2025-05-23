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
experiments = filter(x -> isdir("$(exp_dir)$x") && occursin("_betterEstimVard_", x), readdir(exp_dir));
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
        AEs = getindex.(data, 3);
        DPMLEs = getindex.(data, 4);
        CCEs = getindex.(data, 5);

        meanSOMLEs = zeros(sizeD);
        stdSOMLEs = zeros(sizeD);
        meanAEs = zeros(sizeD);
        stdAEs = zeros(sizeD);
        meanDPMLEs = zeros(sizeD);
        stdDPMLEs = zeros(sizeD);
        meanCCEs = zeros(sizeD);
        stdCCEs = zeros(sizeD);
        for i in 1:sizeD
            valSOMLE = permutedims(cat(SOMLEs[i,:]..., dims=2), (2,1));
            normSOMLEs = sqrt.(sum(valSOMLE.^2,dims=2)[:,1]);
            meanSOMLEs[i] = mean(normSOMLEs);
            stdSOMLEs[i] = std(normSOMLEs);

            valAE = permutedims(cat(AEs[i,:]..., dims=2), (2,1));
            normAEs = sqrt.(sum(valAE.^2,dims=2)[:,1]);
            meanAEs[i] = mean(normAEs);
            stdAEs[i] = std(normAEs);

            valDPMLE = permutedims(cat(DPMLEs[i,:]..., dims=2), (2,1));
            normDPMLEs = sqrt.(sum(valDPMLE.^2,dims=2)[:,1]);
            meanDPMLEs[i] = mean(normDPMLEs);
            stdDPMLEs[i] = std(normDPMLEs);

            valCCE = permutedims(cat(CCEs[i,:]..., dims=2), (2,1));
            normCCEs = sqrt.(sum(valCCE.^2,dims=2)[:,1]);
            meanCCEs[i] = mean(normCCEs);
            stdCCEs[i] = std(normCCEs);
        end

        cblind_colors = palette(:seaborn_colorblind6);
        dict_colors = Dict("SO" => cblind_colors[1],
                           "AE" => cblind_colors[3],
                           "DP" => cblind_colors[4],
                           "CCE" => cblind_colors[2]);

        # Plot Loss
        plot(rangeDims, meanSOMLEs, ribbon=stdSOMLEs, fillalpha=.3, label="SO", linestyle=:auto, color=dict_colors["SO"], linewidth=2, legend=:topleft, xlabel=L"d", ylabel=L"||\widehat \theta_n - \theta^\star||_{2}", guidefontsize=18, tickfontsize=16, legendfontsize=12);
        plot!(rangeDims, meanAEs, ribbon=stdAEs, fillalpha=.3, label="AE", linestyle=:auto, color=dict_colors["AE"], linewidth=2);
        plot!(rangeDims, meanDPMLEs, ribbon=stdDPMLEs, fillalpha=.3, label="DP", linestyle=:auto, color=dict_colors["DP"], linewidth=2);
        plot!(rangeDims, meanCCEs, ribbon=stdCCEs, fillalpha=.3, label="CCE", linestyle=:auto, color=dict_colors["CCE"], linewidth=2);
        file_plot = "$(exp_dir)$(experiment)/plot_betterEstimvard_inst_Gaussian_loss.$format";
        savefig(file_plot);
        
    else
        println("Plots in .$format based on $(exp_dir)$(experiment) already exist.");
    end
    println("");
end


