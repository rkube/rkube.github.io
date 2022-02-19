---
layout: post
title: "Backpropagation through numerical calculations"
date: 2022-02-16 07:00:00 -0400
categories: julia zygote
use_math: true
math_engine: mathjax
---

Differentiable programming makes all code amenable to gradient based
optimization. This harmless statement has broad implications. Think for 
example about a physical simulation where a differential equation is
integrated in time. Given some initial conditions, boundary conditions,
and parameters, the simulator approximates how the system evolves in time.
Now lets say you simulate a wave in a swimming pool that moves towards
a wall. Your goal is to find out how fast the wave has to be launched as
to swap over the edge it is travelling to. With differentiable programming,
you could define a target metric and optimize it with respect to the
initial condition, here the speed with which the wave is launched.
