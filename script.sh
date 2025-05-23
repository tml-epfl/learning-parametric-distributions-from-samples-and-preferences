#!/usr/bin/env bash

# Bash settings
set -o errexit    # abort on nonzero exitstatus
set -o pipefail   # don't hide errors within pipes
set -o nounset    # abort on unbound variable

# Variables
readonly script_name=$(basename "${0}")                                   # Name of the script
readonly script_dir=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )     # Directory of the script
IFS=$'\t\n'                                                               # Split on newlines and tabs (but not on spaces)
NCPU="${1:-4}"                                                            # Number of CPUs used by Julia

# Description of the script
echo -n "Running the script ${script_name} in ${script_dir}."
printf "\n"
echo -n "Each Julia job will use ${NCPU} CPUs."
printf "\n\n"

# Univariate Gaussian: Figure 1(a)
echo -n "Running: julia -O3 -p${NCPU} gaussian1d.jl --expe 'error1d' --instance 'Gaussian' --seed 42 --Nruns 1000 --Nsteps 10000 --batch 100 --sizemax 100"
printf "\n\n"
julia -O3 -p"${NCPU}" gaussian1d.jl --expe "error1d" --instance "Gaussian" --seed 42 --Nruns 1000 --Nsteps 10000 --batch 100 --sizemax 100
printf "\n\n"

# Multivariate Gaussian: Figure 1(b)
echo -n "Running: julia -O3 -p${NCPU} gaussianlarged.jl --expe 'errorlarged' --instance 'Gaussian' --dimension 20 --seed 42 --Nruns 100 --Nsteps 10000 --batch 400 --sizemax 25"
printf "\n\n"
julia -O3 -p"${NCPU}" gaussianlarged.jl --expe "errorlarged" --instance "Gaussian" --dimension 20 --seed 42 --Nruns 100 --Nsteps 10000 --batch 400 --sizemax 25
printf "\n\n"

# Varying dimension for Gaussian: Figure 2
echo -n "Running: julia -O3 -p${NCPU} gaussianvard.jl --expe 'errorvard' --instance 'Gaussian' --dimMax 100 --dimStep 10 --seed 42 --Nruns 1000--Nsteps 10000"
printf "\n\n"
julia -O3 -p"${NCPU}" gaussianvard.jl --expe "errorvard" --instance "Gaussian" --dimMax 100 --dimStep 10 --seed 42 --Nruns 1000 --Nsteps 10000
printf "\n\n"

# Asymptotic Gaussian variance gap: Figure 3
echo -n "Running: julia -O3 -p${NCPU} gapsD.jl --expe 'Matrices' --instance 'Gaussian' --dimMax 100 --seed 42 --Nruns 1000000"
printf "\n\n"
julia -O3 -p"${NCPU}" gapsD.jl --expe "Matrices" --instance "Gaussian" --dimMax 100 --seed 42 --Nruns 1000000 
printf "\n\n"

# Laplace: Figure 4(a)
echo -n "Running: julia -O3 -p${NCPU} gaussian1d.jl --expe 'error1d' --instance 'Gaussian' --seed 42 --Nruns 1000 --Nsteps 10000 --batch 100 --sizemax 100"
printf "\n\n"
julia -O3 -p"${NCPU}" laplace.jl --expe "error1d" --instance "Laplace" --seed 42 --Nruns 10 --Nsteps 10000 --batch 100 --sizemax 100
printf "\n\n"

# Rayleigh: Figure 4(b)
echo -n "Running: julia -O3 -p${NCPU} rayleigh.jl --expe 'error1d' --instance 'Rayleigh' --seed 42 --Nruns 100 --Nsteps 10000 --batch 100 --sizemax 100"
printf "\n\n"
julia -O3 -p"${NCPU}" rayleigh.jl --expe "error1d" --instance "Rayleigh" --seed 42 --Nruns 100 --Nsteps 10000 --batch 100 --sizemax 100
printf "\n\n"

# Other estimators univariate Gaussian: Figure 5(a)
echo -n "Running: julia -O3 -p${NCPU} betterEstim.jl --expe 'betterEstim' --instance 'Gaussian' --seed 42 --Nruns 100 --Nsteps 10000 --batch 100 --sizemax 100"
printf "\n\n"
julia -O3 -p"${NCPU}" betterEstim.jl --expe "betterEstim" --instance "Gaussian" --seed 42 --Nruns 100 --Nsteps 10000 --batch 100 --sizemax 100
printf "\n\n"

# Other estimators multivariate Gaussian: Figure 5(b)
echo -n "Running: julia -O3 -p${NCPU} betterEstimLargeD.jl --expe 'betterEstimlarged' --instance 'Gaussian' --seed 42 --dimension 20 --seed 42 --Nruns 100 --Nsteps 10000 --batch 400 --sizemax 25"
printf "\n\n"
julia -O3 -p"${NCPU}" betterEstimLargeD.jl --expe "betterEstimlarged" --instance "Gaussian" --seed 42 --dimension 20 --seed 42 --Nruns 100 --Nsteps 10000 --batch 400 --sizemax 25
printf "\n\n"

# Other estimators varying dimension for Gaussian: Figure 6
echo -n "Running: julia -O3 -p${NCPU} betterEstimvard.jl --expe 'betterEstimVard' --instance 'Gaussian' --dimMax 100 --dimStep 10 --seed 42 --Nruns 100 --Nsteps 10000"
printf "\n\n"
julia -O3 -p"${NCPU}" betterEstimvard.jl --expe "betterEstimVard" --instance "Gaussian" --dimMax 100 --dimStep 10 --seed 42 --Nruns 100 --Nsteps 10000
printf "\n\n"

# Other estimators based on convex surrogate of the 0-1 loss: Figure 7
echo -n "Running: julia -O3 -p${NCPU} losses.jl --expe 'losses' --instance 'Gaussian' --seed 42 --Nruns 10 --Nsteps 10000 --batch 100 --sizemax 100"
printf "\n\n"
julia -O3 -p"${NCPU}" losses.jl --expe "losses" --instance "Gaussian" --seed 42 --Nruns 10 --Nsteps 10000 --batch 100 --sizemax 100
printf "\n\n"

# Impact normalization and regularization: Figure 8
echo -n "Running: julia -O3 -p${NCPU} norm_and_regu.jl --expe 'normregu' --instance 'Gaussian' --seed 42 --Nruns 100 --Nsteps 10000 --batch 100 --sizemax 100"
printf "\n\n"
julia -O3 -p"${NCPU}" norm_and_regu.jl --expe "normregu" --instance "Gaussian" --seed 42 --Nruns 100 --Nsteps 10000 --batch 100 --sizemax 100
printf "\n\n"
