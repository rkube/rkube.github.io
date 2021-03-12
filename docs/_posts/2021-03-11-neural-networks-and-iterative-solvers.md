---
layout: post
title: "Machine-learned preconditioners"
date: 2021-03-02 11:56:22 -0400
categories: jekyll update
usemathjax: true
---

A common task in scientific computing is to solve a system of linear equations

$$A x = b $$

with a matrix $$A \in \mathcal{R}^{d\ times d}$$ that gives the coefficients of the 
system, a right-hand-side vector $$b \in \mathcal{R}^{d}$$,
and a solution vector $$x \in \mathcal{R}^{d}$$.


Instead of solving the linear problem directly, one often would like to solve
the preconditioned problem. We write this problem by inserting a one in the
linear system and re-parenthesing:

$$\left( A P^{-1} \right) \left( P x \right) = b$$

In this equation we have changed the coefficient matrix from $$A$$ to
$$A P^{-1}$$ and the solution vector is now $$P x$$ instead of $$x$$. To
retrieve the original vector, simply calculate $$ x = P^{\1}y$$. We call
$$P^{-1}$$ the preconditioner. 


The Gauss-Seidel method is an iterative method to find the solution of
a linear system. Given an initial guess $$x_0$$, update the entries of this
vector following an iterative scheme and after some, or many, iterations, $$x_0$$
solves the equation $$A x_0 = b$$. 
[Wikipedia](https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method) has a nice 
and instructive page on this scheme.

A good preconditioner gives you a converged solution after fewer iterations.
In other words, with a good preconditioner you are closer to the true solution
of the linear system than with a bad preconditioner.
Choosing a good preconditioner depends on the problem at hand and can become
a dark art. Let's make it even more dark and train a neural network to be
a preconditioner. As a first step we can model $$P^{-1}$$ as a
multi-layer perceptron (technically a single-layer):

$$P^{-1} = \sigma \left( W x + \widetilde{b} \right)$$

with a weight matrix $$W \in \mathcal{R}^{d^2 \times d}$$ and a bias
$$\wildetilde{b} \in \mathcal{R}^{d}. With $$x \in \mathcal{R}^{d}$ the
matrix dimensions are chosen such that $$P^{-1} \in \mathcal{R}^{d \times d}$$

Here I'm discussing a simple test case and am working with the Gauss-Seidel
iterative solver. For real problem we would not do that, but this is a 
proof-of-concept. Anyway, for Gauss-Seidel to work, the system coefficient
matrix $$AP^{-1}$$ needs to be positive-definite.  How do we do this for
our neural network? We can do this be considering only small vectors of $$x$$,
with all entries close to zero. Then we still need that $$\widetilde{b}$$ is
positive-definite. We get the last property by letting 
$$\widetilde{b} = b_0 b_0^{T}$$ and sampling the entries of $$b_0$$ as
$$b_{ij} \sim \mathcal{N}(0, 1)$$. In a similar fashion, we sample the entries
of the weight-matrix $$W$$ as $$w_{i,j} \sim \mathcal{N}(0,1)$$.

To make this preconditioner useful, we need to find a good weight and bias 
matrix. We can do this using automatic differentiation like so:

* Pick an initial guess $$x_0$$ with $$x_{0,i} \sim \mathcal{N}(0, 0.01)$$
* Build the preconditioner $$P^{-1} = \sigma(W, x_0, \widetilde{b})$$
* Calculate 5 Gauss-Seidel steps. Let's call the solution here $$y_5$$.
* Un-apply the preconditioner: $$x_5 = P^{-1} y_5$$
* Calculate the distance to the true solution $$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \left( x_{\text{true}, i} - x_{5,i} \right)^2$$
* Calculate the gradients $$\nabla_{W} \mathcal{L} $$ and
$$\nabla_{B} \mathcal{L}$$
* Update the weight matrix and bias using gradient descent: $$W \leftarrow W - \alpha \nabla_{W} \mathcal{L}$$ and $$b \leftarrow b - \alpha \nabla_{W}$$. Here $$\alpha$$ is the learning rate


We need automatic differentiation to calculate $$\nabla_W \mathcal{L}$$ and 
$$\nabla_W \mathcal{b}$$. This can be done using [Zygote](https://fluxml.ai/Zygote.jl/latest/).



Here is the code

```julia

# Test differentiation through control flow

# Use a iterative conjugate solver with preconditioner

using Random
using Zygote
using LinearAlgebra
using Distributions
using NNlib
```

I copy-and-pasted the Gauss-Seidel code from wikipedia:
```julia
function gauss_seidel(A, b, x, niter)
    # This is from https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method
    x_int = Zygote.Buffer(x)
    x_int[:] = x[:]
    for n ∈ 1:niter
        for j ∈ 1:size(A, 1)
            x_int[j] = (b[j] - A[j, :]' * x_int[:] + A[j, j] * x_int[j]) / A[j, j]
        end
    end
    return copy(x_int)
end
``` 


The block below sets things up.
```julia

Random.seed!(1)
dim = 4

# Define a matrix
A0 = [10.0 -1.0 2.0 0.0; -1 11 -1 3; 2 -1 10 -1; 0 3 -1 8]
# Define the RHS
b0 = [6.0; 25.0; -11.0; 15.0]
# This is the true solution
x_true = [1.0, 2.0, -1.0, 1.0]

# Define an initial state. Draw this from a narrow distribution around zero
# We need to do this so that the preconditioner eigenvalues are positive
x0 = rand(Normal(0.0, 0.01), 4)

# Define an MLP. This will later be our preconditioner
# The output should be a matrix and we work with 2dim as the size for the MLP
W = rand(dim*dim, dim)
# For Gauss-Seidel to work, the matrix A*P⁻¹ needs to be positive semi-definite.
bmat = rand(dim, dim)
# We know that any matrix A*A' is positive semi-definite
bvec = reshape(bmat * bmat', (dim * dim))
# Now Wx + bmat is positive semi-definite if x is very small
P(x, W, b) = NNlib.relu.(reshape(W*x .+ b, (dim, dim)))
# Positive-definite means positive Eigenvalues. We should check this.
@show eigvals(A0*P(x0, W, bvec))
```


This function will serve as our loss function. It evaluates the NN preconditioner and then runs
some Gauss-Seidel iterations. Finally, the iterative solution
approximation is transformed back by applying $$P^{-1}$$.

```
function loss_fun(W, bmat, A0, b0, y0, niter=5)
    # W - Weight matrix for NN-preconditioner
    # bmat - Bias vector for NN preconditioner
    # A0: Linear system coefficient matrix
    # b0: RHS of linear system
    # y0 - initial guess for Linear system. Strictly, this is x0. But we call it the same
    # assuming that P⁻¹x0 = x0.
    # niter - Number of Gauss-Seidel iterations to perform
    #
    # Evaluate the preconditioner
    P⁻¹ = P(y0, W, reshape(bmat * bmat', (dim * dim)))
    # Initial guess
    # Now we solve A(Px)⁻¹y = rhs for y with 3 Gauss-Seidel iterations
    y_sol = gauss_seidel(A0 * P⁻¹, b0, y0, niter)
    # And reconstruct x
    x = P⁻¹ * y_sol


    loss = norm(x - x_true) / length(x)
    return loss
end
```

Now comes the fun part. Zygote calculates the partial derivatives of the
loss function with respect to its input, in our case $$W$$ and $$b$$. 
Given the gradients, we can actually update $$W$$ and $$b$$.

```
num_epochs = 10
loss_arr = zeros(num_epochs)
W_arr = zeros((size(W)..., num_epochs))
bmat_arr = zeros((size(bmat)..., num_epochs))
α = 0.05

for epoch ∈ 1:num_epochs
    loss_arr[epoch], grad = Zygote.pullback(loss_fun, W, bmat, A0, b0, x0)
    res = grad(1.0)
    # Gradient descent
    global W -= α * res[1]
    global bmat -= α * res[2]
    # Store W and b
    W_arr[:, :, epoch] = W
    bmat_arr[:, :, epoch] = bmat
end
```

Now we need some code to find out how good we did.
```
# Now evaluate how this performs
# This functions returns a vector with the residual of the iterative
# solution to the true solution at each step
function eval_performance(W, bmat, A0, y0, b0, niter=20)
    # Instantiate the preconditioner with the initial guess
    P⁻¹ = P(y0, W, reshape(bmat * bmat', (dim * dim)))
    y_sol = copy(y0)
    loss_vec = zeros(niter)
    for n ∈ 1:niter      
        # Initial guess
        # Now we solve (Px)⁻¹y = rhs for y with 3 Gauss-Seidel iterations
        y_sol = gauss_seidel(A0 * P⁻¹, b0, y_sol, niter)
        # And reconstruct x
        x = P⁻¹ * y_sol
        loss_vec[n] = norm(x - x_true) / length(x)
    end

    return loss_vec
end

# Get the residual at each Gauss-Seidel iteration
sol_err = zeros(20, num_epochs)
for i ∈ 1:num_epochs
    sol_err[:, i] = eval_performance(W_arr[:, :, i], bmat_arr[:, :, i], A0, x0, b0)
end

```

At a later point I should update this blog-post with some posts.
Currently I'm also training only on a single vector. Performance is too good,
that is, we overfit. So this code should use a larger number of initial
conditions.

