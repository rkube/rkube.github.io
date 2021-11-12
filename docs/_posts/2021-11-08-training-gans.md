---
layout: post
title: "Training GANs in Julia's Flux"
data: 2021-11-07 20:00:00 -0400
categories: julia gan 
use_math: true
math_engine: mathjax
---

In order to effectively run machine learning experiments we need a fast
turn-around time for model training. So simply implementing the model is not
the only thing we need to worry about. We also want to be able to change the
hyperparameters in a convenient way. This could either be through a configuration
file or through command line arguments. This post demonstrates how I train
a [vanilla GAN](https://fluxml.ai/tutorials/2021/10/14/vanilla-gan.html) on the
MNIST dataset. It is not about GAN theory, for this the original paper by
Goodfellow et al. [[1]] is a good starting point. Instead I focus on how to
structure the code and subtle implementation issues I came across when writing
the code. You can find the current version of the code on [github](https://github.com/rkube/mnist_gan).

## Project structure
I am taking a starting point in the vanilla GAN implementation on the 
[FluxML website](https://fluxml.ai/tutorials/2021/10/14/vanilla-gan.html). This
implementation works and the trained generator indeed generates images that
look indistinguishable from images belonging to the MNIST dataset.
But how do we arrive there? Why are the learning rates chosen as $$\eta = 2 \times 10^{-4}$? IS the `leakyrelu` the optimal activation function or does it perform
on-par with `relu` in some regime? To answer these questions we need a code that 
quickly allows us to change these parameters. 

And while we are at it, lets bundle the code together with its dependencies in a
Julia package. This allows us to conveniently a package dependencies to the code.
Taken together, the code and well defined dependencies make the behaviour reproducible.  The Julia documentation gives a comprehensive introduction on
packages [here](https://pkgdocs.julialang.org/v1/creating-packages/).

In order to run the code in the project I first checkout the code from github,
then enter the repository and then execute the runme script:

```julia
$ git checkout https://github.com/rkube/mnist_gan.git
$ cd mnist_gan
$ julia --project=. src/runme.jl --activation=ADAM --train_k=8 ...
```

All packages installed in the project are local to this project and don't interfere
with packages installed in the general environment. This allows for example to
specify for certain version numbers and will give us producibility of our results.


## Code structure
The code is structued as a standard Julia project. The root folder layout looks
like this

```.
├── Manifest.toml
├── Project.toml
├── README.md
└── src
    ├── Manifest.toml
    ├── mnist_gan.jl
    ├── models.jl
    ├── Project.toml
    ├── runme.jl
    └── training.jl
```

The root folder contains `Manifest.toml` and `Project.toml` which include information
about dependencies, versions, package names. More information is given in the
[Pkg.jl documentation](https://pkgdocs.julialang.org/v1/toml-files/).

The `src` folder contains all source codes files. In particular it contains a
`mnist_gan.jl` file. This is named after the package name and in the simple case
here only twofines the package as a module, includes all other modules and
my two source files

```julia
module mnist_gan

using NNlib
using Flux
using Zygote
using Base:Fix2

# All GAN models
include("models.jl")
# Functions used for training
include("training.jl")
end #module
```

As additional structure I put the models in `models.jl` and training functions in
`training.jl`.

## Command line arguments
To quickly train the GAN with specific hyperparameters one can either read the
hyperparameters from a configuration file or pass them through the command line.
Here we do the second approach. To comfortably parse command line arguments I'm
using (ArgParse.jl)[https://argparsejl.readthedocs.io/en/latest/argparse.html].
 
Condensing to only single argument, my code looks like this:
```julia
using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
    "--lr_dscr"
        help = "Learning rate for the discriminator. Default = 0.0002"
        arg_type = Float64
        default = 0.0002

args = parse_args(s)
```
That's it. Now I can access the single command line arguments via `args[lr_dscr]`.


## Logging
Keeping track of the model performance while training is crucial when performing
parameter scans. For the vanilla GAN alone I defined 10 parameters that can be
varied. Letting each parameter assume only two distinct values this allows for
1024 combinations. Julia's (logging facilities)[https://github.com/JuliaLogging]
provide means to systematicallylog model training for a large hyperparameter scan.

In particular, we can use (TensorBoardLogger.jl)[https://github.com/JuliaLogging/TensorBoardLogger.jl]. (TensorBoard)[https://www.tensorflow.org/tensorboard]
provides a visualization of training and includes numerous useful features, such
as visualization of loss curves, displaying of model output images and more. To
use TensorBoardLogger.jl in my code I have to include the module, instantiate
a logger. Then I can easily log my experiments:

```julia
# Import the modules
using TensorBoardLogger
...
# Instantiate TensorBoardLogger
# Let's log the hyperparamters of the current run. 
dir_name = join(["$(k)_$(v)" for (k, v) in a])
tb_logger = TBLogger("logs/" * dir_name)
with_logger(tb_logger) do
    @info "hyperparameters" args
end

# Wrap the main training loop in a with clause to enable logging
lossvec_gen = zeros(Float32, args["num_iterations"])
lossvec_dscr = zeros(Float32, ["num_iterations"])

with_logger(tb_logger) do
    for n ∈ args["num_iterations"]e
        # Do machine learning ...
        ...
        # Code to log PNG images to tensorboard, inside the main training loop
        if n % args["output_period"] == 0
            noise = randn(args["latent_dim"], 4) |> gpu;
            fake_img = reshape(generator(noise), 28, 4*28) |> cpu;
            # I need to clip pixel values to [0.0; 1.0]
            fake_img[fake_img .> 1.0] .= 1.0
            fake_img[fake_img .< -1.0] .= -1.0
            fake_img = (fake_img .+ 1.0) .* 0.5
            # 
            log_image(tb_logger, "generatedimage", fake_img, ImageFormat(202))
        end
        # Log the generato and discriminator loss
        @info "test" loss_generator=lossvec_gen[n] loss_discriminator=lossvec_dscr[n]
    end # for
end #  Logger
```

First, I'm generating a string from all keys and values defined in the command
line argument dictionary. Later this will allow me to filter these arguments.
Then I'm logging the `args` dictionary, which contains the hyperparameters of
the current experiment. Then I'm generating a fake image using the generator
and log it as well. Here I need to clip the pixel values to [0.0; 1.0]. Since
the Generator is trained on images with pixel values between -1.0 and 1.0 I need
to transform the pixel space. Note that he last argument to the call in 
`log_image` encodes the layout of the `fake_img` array. I had to look up the
available encodings via

```julia
?
```

## Loss functions on-the-fly
To resolve the correct loss function from command line arguments I'm using the
getfield method. To make it a little more convoluted, we also need to distinguish
between loss functions that take an additional, tunable parameterr
like `celu`, `elu`, `leakyrelu` and `trelu`,  and loss functions who do not.
The following code block shows how to map a string that encodes the function name
to the actual function using `getfield`. To create a closure over an optional
parameter I'm using `Fix2`. The code below is from `models.jl`

```julia
function get_vanilla_discriminator(args)
    ...
    if args["activation"] in ["celu", "elu", "leakyrelu", "trelu"]
        # Now continue: We want to use Base.Fix2
        act = Fix2(getfield(NNlib, Symbol(args["activation"])), Float32(args["activation_alpha"]))
    else
        act = getfield(NNlib, Symbol(args["activation"]));
    end

    return Chain(Dense(28 * 28, 1024, act), 
        ...);
```

I found out that can have an impact on performance how I pass the activation
function as an argument to the dense layer. By passing only the function, the
implementation of Dense handles how the activation function is applied to the linear
transformation. This is how it should be. If I manually prescribe how to apply
the broadcast I find slower performance:

```julia
julia> d1 = Dense(100, 100, act)
Dense(100, 100, relu)  # 10_100 parameters

julia> @btime d1(randn(Float32, 100, 100));
  163.863 μs (6 allocations: 117.33 KiB)

julia> d2 = Dense(100, 100, x -> act(x))
Dense(100, 100, #5)  # 10_100 parameters

julia> @btime d2(randn(Float32, 100, 100));
  3.041 ms (20016 allocations: 430.30 KiB)
```

So manually prescribing how to perform the broadcast is about 20 times slower.
Instead, I let the code above return a function that Flux knows how to apply a 
broadcast on.

## Running a parameter scan
Now we are set up to run a parameter scan. For this I generate runscripts
where I vary my command line arguments. The resulting scripts look like this

```
#SBATCH things

cd /location/of/the/repo
julia --project=. --lr_dscr=0.0002 --lr_gen=0.0002 --batch_size=8 --num_iterations=10000 --latent_dim=100 --optimizer=ADAM --activation=leakyrelu --activation_alpha=0.2 --train_k=8 --prob_dropout=0.3 --output_period=250
```

Of course the arguments vary across the scripts. After crunching all the numbers,
the log file directory is populated with the tensorboard log files. The next
blog post will discuss how the results look like and how to pick the best
hyperparameters.


## References
<a id="1">[1]</a>
I. Goodfellow et al. [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)