# Learning Parametric Distributions from Samples and Preferences

**Marc Jourdan**, **Gizem Yüce**, **Nicolas Flammarion**

**ICML 2025**

## Abstract

Recent advances in language modeling have underscored the role of preference feedback in enhancing model performance. This paper investigates the conditions under which preference feedback improves parameter estimation in classes of continuous parametric distributions. In our framework, the learner observes pairs of samples from an unknown distribution along with their relative preferences depending on the same unknown parameter. We show that preferences-based M-estimators achieve a better asymptotic variance than sample-only M-estimators, further improved by deterministic preferences. Leveraging the hard constraints revealed by deterministic preferences, we propose an estimator achieving an estimation error scaling of $\mathcal{O}(1/n)$---a significant improvement over the $\Theta(1/\sqrt{n})$ rate attainable with samples alone. Next, we establish a lower bound that matches this accelerated rate; up to dimension and problem-dependent constants. While the assumptions underpinning our analysis are restrictive, they are satisfied by notable cases such as Gaussian or Laplace distributions for preferences based on the log-probability reward.

## Getting started

**Install Julia**
```
wget https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.3-linux-x86_64.tar.gz
tar zxvf julia-1.11.3-linux-x86_64.tar.gz
export PATH="$PATH:/path/to/<Julia directory>/bin"
```

**Install packages**
```
julia
using Pkg;
Pkg.add(["JLD2", "Printf", "JSON", "Dates", "IterTools", "Distributed", "JuMP", "Ipopt", "HiGHS"]);
Pkg.add(["Random", "LinearAlgebra", "Distributions", "CPUTime"]);
Pkg.add(["StatsPlots", "ArgParse", "Statistics", "StatsBase", "Plots"]);
Pkg.add(["Pickle", "ColorSchemes", "Distributed", "LaTeXStrings"]);

using ArgParse, JLD2, Printf, JSON, Dates, IterTools, Random, CPUTime, Pickle;
using LinearAlgebra, Statistics, Distributions, StatsBase, LaTeXStrings;
using Plots, StatsPlots, ColorSchemes;
using JuMP, Ipopt, HiGHS;
using Distributed;
```

## Run experiments

After installing Julia, to run the experiments presented in the paper, you can either use the custom commands (defined below) to only perform a given experiment or you can directly run the `script.sh`. Some experiments are computationally costly, hence we recommend to use more cores than solely four (`-p4`).

```
cd path_to_folder/code
mkdir experiments
mkdir data
```

### All experiments at once

To run our script with four cores:
```
chmod +x script.sh
bash script.sh 4
```
Note that the script doesn't include the plotting functions. Therefore, additional runs of the corresponding plotting functionalities have to be made afterwards (see below). As it is, you might need to be careful in the order in which you call the visualization functions since the plots are created for all the folders in the `experiments` folder, which don't have `.pdf` inside them. Also, better visualization can be obtained by commenting some lines of codes and replacing them with others.

### One at a time

#### Univariate Gaussian: Figure 1(a)

```
cd path_to_folder/code
julia -O3 -p4 gaussian1d.jl --expe "error1d" --instance "Gaussian" --seed 42 --Nruns 1000 --Nsteps 10000 --batch 100 --sizemax 100
julia viz_gaussian1d.jl --format "svg"
```

#### Multivariate Gaussian: Figure 1(b)

```
cd path_to_folder/code
julia -O3 -p4 gaussianlarged.jl --expe "errorlarged" --instance "Gaussian" --dimension 20 --seed 42 --Nruns 100 --Nsteps 10000 --batch 400 --sizemax 25
julia viz_gaussianlarged.jl --format "svg"
```

#### Varying dimension for Gaussian: Figure 2

```
cd path_to_folder/code
julia -O3 -p4 gaussianvard.jl --expe "errorvard" --instance "Gaussian" --dimMax 100 --dimStep 10 --seed 42 --Nruns 1000 --Nsteps 10000
julia viz_gaussianvard.jl --format "svg"
```

#### Asymptotic Gaussian variance gap: Figure 3

```
cd path_to_folder/code
julia -O3 -p4 gapsD.jl --expe "Matrices" --instance "Gaussian" --dimMax 100 --seed 42 --Nruns 1000000 
julia viz_gapsD.jl --format "svg"
```

#### Laplace: Figure 4(a)

```
cd path_to_folder/code
julia -O3 -p4 laplace.jl --expe "error1d" --instance "Laplace" --seed 42 --Nruns 10 --Nsteps 10000 --batch 100 --sizemax 100
julia viz_laplace.jl --format "svg"
```

#### Rayleigh: Figure 4(b)

```
cd path_to_folder/code
julia -O3 -p4 rayleigh.jl --expe "error1d" --instance "Rayleigh" --seed 42 --Nruns 100 --Nsteps 10000 --batch 100 --sizemax 100
julia viz_rayleigh.jl --format "svg"
```

#### Other estimators univariate Gaussian: Figure 5(a)

```
cd path_to_folder/code
julia -O3 -p4 betterEstim.jl --expe "betterEstim" --instance "Gaussian" --seed 42 --Nruns 100 --Nsteps 10000 --batch 100 --sizemax 100
julia viz_betterEstim.jl --format "svg"
```

#### Other estimators multivariate Gaussian: Figure 5(b)

```
cd path_to_folder/code
julia -O3 -p4 betterEstimLargeD.jl --expe "betterEstimlarged" --instance "Gaussian" --seed 42 --dimension 20 --seed 42 --Nruns 100 --Nsteps 10000 --batch 400 --sizemax 25
julia viz_betterEstimLarged.jl --format "svg"
```

#### Other estimators varying dimension for Gaussian: Figure 6

```
cd path_to_folder/code
julia -O3 -p4 betterEstimvard.jl --expe "betterEstimVard" --instance "Gaussian" --dimMax 100 --dimStep 10 --seed 42 --Nruns 100 --Nsteps 10000
julia viz_BetterEstimVard.jl --format "svg"
```

#### Other estimators based on convex surrogate of the 0-1 loss: Figure 7

```
cd path_to_folder/code
julia -O3 -p4 losses.jl --expe "losses" --instance "Gaussian" --seed 42 --Nruns 10 --Nsteps 10000 --batch 100 --sizemax 100
julia viz_losses.jl --format "svg"
```

#### Impact normalization and regularization: Figure 8

```
cd path_to_folder/code
julia -O3 -p4 norm_and_regu.jl --expe "normregu" --instance "Gaussian" --seed 42 --Nruns 100 --Nsteps 10000 --batch 100 --sizemax 100
julia viz_normregu.jl --format "svg"
```


## Citation

If you find this work useful in your own research, please consider citing it: 
```bibtex
@article{jourdan2025learningwithpreference,
      title={Learning Parametric Distributions from Samples and Preferences}, 
      author={Jourdan, Marc and Y{\"u}ce, Gizem and Flammarion, Nicolas},
      journal={International Conference on Machine Learning (ICML)},
      year={2025}
}
```

### License
This codebase is released under [MIT License](LICENSE).