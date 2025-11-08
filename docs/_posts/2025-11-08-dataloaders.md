---
layout: post
title: "A Beginner's Guide to DataLoaders in Julia with MLUtils.jl"
date: 2025-11-08 00:00:00 -0400
categories: julia deep-learning dataloaders
use_math: true
math_engine: mathjax
---

# Introduction

Data Loaders are critical tools to work efficiently with deep learning.
They are the glue that bring the data to your machine learning model. Doing this efficiently
and without errors is important for successful training of any model.

Datasets and DataLoaders are concepts that encapsulate the logic for preparing the data to be
input in your model. In their basic form, they prepare small batches of data for ingestion by
reshaping individual samples into matrix form. More involved applications may load 
images from files, augment them through flipws, crops, rotations. 

The good news is that in Julia, [MLUtils](https://juliaml.github.io/MLUtils.jl) has you covered for 
all your data loading needs. In this blog post we'll dive into how to create both Datasets and DataLoaders.
We start out with a simple-as-possible example and then walk through a more realistic example.

# Data loaders in easy mode
Let's start by writing a simple as possible data loader to illustrate the core concepts of data loading
using the DataLoader. The task we are looking at is somewhat related to image classification, but simplified.

Our data may be a bunch of images, like 1,000 of them. Each of the images is assumed to be in one of
ten classes. A mock up of that data is something like

```julia
X_train = randn(Float32, 10, 10, 1_000)
Y_train = randn(1:10, 1_000)
```

Here, `X_train` are 1_000 samples of 10x10 matrices. These are mocked up as normally distributed Float32
data with shape (10, 10, 1_000), corresponding to (width, height, sample). Julia stores arrays in column-major,
so in our case the data for width is consecutive in memory. The last dimension, the sample dimension varies slowest.
The targets are mocked as 10_000 samples of random integers between 1 and 10, indicated by the `1:10` syntax.

A nice thing about Julia is that you often get away without defining custom classes. In our simlpe case,
we can get away with defining our dataset as 
```julia
data = (X_train, Y_train)
```

No custom class definition needed. The dataset is just a tuple of two vectors.

Now we move on to the data loader. The job of the data loader is to sample from our dataset. It should
do batching, i.e. loading multiple (x,y) samples in one go, and random shuffling.
For this simple dataset, we just need to create a DataLoader:
```julia
loader = DataLoader(data, batch_size = 3)
```

DataLoaders are iterable, and we can look at the first item like so:
```julia
julia> (x_first, y_first) = first(loader);
julia> size(x_first)
(10, 10, 3)
julia> size(y_first)
(3,)
```

When we iterate over the dataloader, we can use individual samples. Here is the iteration skeleton:
```julia
julia> for (x, y) in loader
            @show size(x), size(y)
       end
[...]
(size(x), size(y)) = ((10, 10, 3), (3,))
(size(x), size(y)) = ((10, 10, 1), (1,))
```

Note that in the last iteration, the DataLoader only returns vectors with a single sample.
This is the loader watching out for us and stopping at the end of the data. We have 1_000 samples,
and using `batchsize=3`, the last iteration can only feature a single sample. This is because
1000 / 3 = 333.33333 and the dataset is exhausted after the first sample in the last iteration.

We can confirm this by querying the length of the dataloader:
```
julia> length(loader)
334
```

That's it. In conclusion, for a simple in-memory dataset creating a tuple `data = (X, Y)` is all you need.
The DataLoader does the rest.

# A more realistic example
Now let's look at a more involved case where we want to read text data from a file and tokenize it.
This sample is inspired by Andrej Karpathy's [NanoGPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY),
[github](https://github.com/karpathy/ng-video-lecture) [collab](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing).

For this application, it's convenient to encapsulate information on the DataSet into a struct. This struct
collects the length of the text, a block size for data loading, dictionaries that map characters to tokens
and vice-versa, and a vector of all the tokens. The members important for the data loading logic are 
`block_size` and `data`. While `data` holds the data vector itself, `block_size` defines the length of
an individual sample.

In addition, we give the struct a constructors that reads all lines in the text file, creates an array of
unique characters, and dictionaries to map from character to token and a dictionary to map from token to character.
While the dictionaries are not necessary for the data loading logic to work, they are included here for completeness.

```
struct NanoDataset
    block_size::Int64               # How long an observation is. This is important for the dataloader
    ch_to_int::Dict{Char, Int64}    # Maps chars to tokens
    int_to_ch::Dict{Int64, Char}    # Maps tokens to chars
    data::Vector{Int64}             # The tokens

    function NanoDataset(filename::String, block_size::Int64)
        lines = readlines(filename)             # Read all lines
        _out = [c for l ∈ lines for c ∈ l]      # Create char array
        chars = sort!(unique(_out))             # Sort all chars occuring in the dataset
        push!(chars, '\n')                      # Add the \n character that we stripped when only looking at lines
        ch_to_int = Dict(val => ix for (ix, val) in enumerate(chars))   # Mapping of chars to int
        int_to_ch = Dict(ix => val for (ix, val) in enumerate(chars))


        all_tokens = [ch_to_int[s] for s in join(lines, "\n")]

        new(block_size, ch_to_int, int_to_ch, all_tokens)
    end
end
```

To create this dataset we run

```julia
ds = NanoDataset(FILENAME, 16)
```

Great, now moving on to the DataLoader. To make the dataloader work, we have to implement the `numobs` and
`getobs` interface as described in the 
[Documentation](https://juliaml.github.io/MLUtils.jl/stable/api/#MLUtils.DataLoader). This basically means
that we have to make DataLoader know how to get the length of the dataset and how to get a single observation.


Fortunately, the MLUtils documentation tells us how to implement both. [numobs](https://juliaml.github.io/MLUtils.jl/stable/api/#MLCore.numobs) just requires to specialize `Base.length` for our type. This function returns how
many observations (that is the number of samples) there are in the dataset.

We implement this like
```
Base.length(d::NanoDataset) = length(d.data) - d.block_size - 1
```

which makes `numobs` work for our dataset:

```julia
julia> numobs(ds)
1115375
```
Next, we'll get `getobs` working. 


For the easy case, an individual sample of `X_train` can be accessed by `X_train[:,:,42]`. That is, array 
indexing is the access. For `NanoDataset` we need to define how to get a single observation. We do this
by specializing `Base.getindex` on our dataset:

```
function Base.getindex(d::NanoDataset, i::Int)
    1 <= i <= length(d) - d.block_size - 1|| throw(ArgumentError("Index is out of bounds"))
    return (d.data[i:i+d.block_size-1], d.data[i+1:i+d.block_size])
```

So when we access ds through `[]` braces, we get two vectors: The input `X` and the target `Y`. `X` is 
a sequence of tokens of length `block_size` and `Y` is this same sequence, shifted by a single index.

I'd like to note here, that `Base.getindex` is a good place to get fancy. For image loading, this
function may load data from a file. Or apply augmentations through 
[Augmentor.jl](https://github.com/Evizero/Augmentor.jl). The sky (or ChatGPT) is the limit!


Now we can access samples in our dataset like

```
julia> ds[1]
([18, 47, 56, 57, 58, 1, 15, 47, 58, 47, 64, 43, 52, 10, 65, 14], [47, 56, 57, 58, 1, 15, 47, 58, 47, 64, 43, 52, 10, 65, 14, 43])
```
Note that Y is just X shifted by one index, just as implemented in `getindex`.
Before we define a dataloader, let's implement one more function that allows us to load minibatchs:
```
MLUtils.getobs(d::NanoDataset, i::AbstractArray{<:Integer}) = [getobs(d, ii) for ii in i]
```

This function defines to dispatch calls where requested samples are given by `i::AbstractArray{<:Integer}`
to multiple calls of `getobs`. For example, loading the first two samples returns this:

```julia
julia> getobs(d, [1,2])
2-element Vector{Tuple{Vector{Int64}, Vector{Int64}}}:
 ([18, 47, 56, 57, 58, 1, 15, 47, 58, 47, 64, 43, 52, 10, 65, 14], [47, 56, 57, 58, 1, 15, 47, 58, 47, 64, 43, 52, 10, 65, 14, 43])
 ([47, 56, 57, 58, 1, 15, 47, 58, 47, 64, 43, 52, 10, 65, 14, 43], [56, 57, 58, 1, 15, 47, 58, 47, 64, 43, 52, 10, 65, 14, 43, 44])

```

Fantastic, now we have everything in place to create a DataLoader. We can create a dataloader that
loads 2 batches at the same time: 

```julia
julia> dl = DataLoader(d, batchsize=2)
557688-element DataLoader(::NanoDataset, batchsize=2)
  with first element:
  2-element Vector{Tuple{Vector{Int64}, Vector{Int64}}}

julia> (x,y) = first(dl)
2-element Vector{Tuple{Vector{Int64}, Vector{Int64}}}:
 ([18, 47, 56, 57, 58, 1, 15, 47, 58, 47, 64, 43, 52, 10, 65, 14], [47, 56, 57, 58, 1, 15, 47, 58, 47, 64, 43, 52, 10, 65, 14, 43])
 ([47, 56, 57, 58, 1, 15, 47, 58, 47, 64, 43, 52, 10, 65, 14, 43], [56, 57, 58, 1, 15, 47, 58, 47, 64, 43, 52, 10, 65, 14, 43, 44])

```

The data loader returns a vector of lenth 2, each element of the vector a tuple of vectors.
Ideally for deep learning, we'd like a matrix of size (16,2) though. Remember the first dimension is the sequence
length, dimensions 2 is the batch dimension. The DataLoader can do this by setting `collate=true`:

```julia
julia> dl = DataLoader(d, batchsize=2, collate=true)
557688-element DataLoader(::NanoDataset, batchsize=2, collate=Val{true}())
  with first element:
  (16×2 Matrix{Int64}, 16×2 Matrix{Int64},)

julia> (x,y) = first(dl)
([18 47; 47 56; … ; 65 14; 14 43], [47 56; 56 57; … ; 14 43; 43 44])

julia> x
16×2 Matrix{Int64}:
 18  47
 47  56
 56  57
 57  58
 58   1
  1  15
 15  47
 47  58
 58  47
 47  64
 64  43
 43  52
 52  10
 10  65
 65  14
 14  43
```

Now the DataLoader returns the desired matrix. In addition, DataLoader supports shuffling and multithreading,
just add `shuffle=true, parallel=true` to the parameter list.

# Where to go from here

In this tutorial we looked at how to use DataLoaders in Julia through two examples. In the first example,
we had in-memory data that can be passed as a tuple into the DataLoader. That just worked.
In the second example, we had to implement the `numobs`, `getobs` interface to make it work. This is a
just a bit more work, but allows to work with arbitrary custom data. The `getobs` implementation is where
to hook in lazy-loading from disk, augmentations, and so on.

Now that we looked at some examples on data loading, here are some things to try next
* Try building your own Dataloader and use it in a [Lux](https://lux.csail.mit.edu/) or [Flux](https://fluxml.ai/) training loop.
* Play around with image augmentation using [Augmentor](https://evizero.github.io/Augmentor.jl/stable/)
* Try using threads and measure the speed-up compared to single-threaded loading.






