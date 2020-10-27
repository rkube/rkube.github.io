---
layout: post
title:  "Hyperparameter tuning on traverse using raytune"
date:   2020-10-13 13:54:38 -0400
categories: jekyll update
---


Hyperparameter tuning is crucial to optimize the performance of deep neural networks. While
simple models may only have a few hyperparmaters, more complex network architectures can have
quiet a few. Furthermore, it can also take a while to train a complex neural network on the dataset
at hand. The task of hyperparameter tuning is to identify a set of hyperparameters for which
a given network architecture performs optimal, as measured by a metric such as the loss function. 

For small models, such as a 3-layer multilayer perceptrons, that is used for image classification,
the number of hyperparameters is managable. In this example, we may wish to change 

* The number of units in the middle layer (the number of units in the input and output layer is fixed)
* The initial learning rate
* The optimizer, f.ex. Stochastic gradient descent, ADAM, etc. 
* If we use batch learning we could change the batch-size.

Hyperparameters are distinct from the parameters of a machine learning model in that they are set
prior to learning by the user and will not be optimized during learning.

So, how do we find the optimal hyperparmaters? Common approaches are methods like grid search
and random guessing. Both these methods work by choosing a random combination of
hyperparameters and train the model, measuring the objective function for the given selection.
But while grid search samples the continuous hyperparameters, such as the learning rate,
at equi-distant points (either linspace or logspace), random search would do so at random
points (f.ex. by sampling from a uniform distribution on an interval bounded by max and min
learning rate). The reasoning behind random search's approach is grounded on the fact that
this is a non-smooth optimization problem - the manifold of hyperparameters is not smooth.
By randomly sampling one may discover a set of parameters closer to the global minimum
than by using a regularly spaced grid. But in practice, actually finding a global minimum
can be considered pure luck. Both, grid search and random search require a large amount
of training cycles.

Multiple libraries can help machine learning practitioners in hyperparater tuning. An open-source
library that works well in cluster environment is [ray](https://www.ray.io). Major selling points
for ray are

* It works with most major deep learning frameworks, such as pytorch, tensorflow, mxnet etc.
* It allows to seamlessly command distributed resources
* It implements common optimization algorithms, such as random search or Bayes optimization

The [ray documentation](https://docs.ray.io/en/latest/tune/tutorials/overview.html) 
features a large list of examples and tutorials on how to run ray. In the following, I describe
how I use ray on a slurm cluster.

## Installing ray

As a pre-requisite to run ray on a slurm cluster we need a conda environment with ray installed.
On intel architectures, ray can be installed using pip. On Power9-based clusters ray needs to be
compiled from source. This is how I did it:


1. Clone the git repository since we will be compiling from code:

```
git clone git@github.com:ray-project/ray.git 
```

2. Next I had to install rust as a dependency for py-spy:
```
 curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh 
```

3. Add cargo to my PATH: 
```
$ export PATH=$PATH/.cargo/bin 
```

4. Install py-spy, see this [github issue](https://github.com/benfred/py-spy/issues/277  )
```
cargo install py-spy 
```

5. Then manually remove py-spy from requirements.txt and setup.py in the git repository.

6. To build ray I also had to install the [bazel] (https://bazel.build) build system. I was stuck
installing the Java Development Kit on Power9, but our cluster support solved the issue. Please,
don't use bazel for your work.

7. Now I need to install the javascript package manager npm for the dashboard. Without the dashboard I ran into a problem where ray would not start properly
 as described in this [github issue](https://github.com/ray-project/ray/issues/10803). So heading [over to](https://nodejs.org/en/download/)
 and unpack the LTS version of nodejs into a directory. There we get a full structure of nodejs/bin nodejs/share etc. Add the ...nodejs/bin to your $PATH to compile the dashboard.

```
pushd ray/python/ray/dashboard/client 
npm ci 
npm run build 
popd 
```

8. Now, compile ray. Ensure that the target conda environment is active:
```
$ conda activate ml
$ cd ray/python 
$ pip install -e . --verbose  # Add --user if you see a permission denied error. 
```
 
I had to comment out the flag --experimental_ui_deduplicate (removed in bazel-3.5.0 the newest version) in ray/.bazelrc to get it to compile 

 
## Running ray on a cluster

With ray installed, let's deploy a ray cluster on our slurm-managed cluster. Let's use the script provided in the 
[ray documentation](https://docs.ray.io/en/master/cluster/slurm.html?highlight=slurm#deploying-on-slurm)

```
#!/bin/bash
#SBATCH --job-name=test
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=1GB
#SBATCH --nodes=4
#SBATCH --tasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --reservation=test

let "worker_num=(${SLURM_NTASKS} - 1)"

# Define the total number of CPU cores available to ray
let "total_cores=${worker_num} * ${SLURM_CPUS_PER_TASK}"

suffix='6379'
ip_head=`hostname`:$suffix
export ip_head # Exporting for latter access by trainer.py

# Start the ray head node on the node that executes this script by specifying --nodes=1 and --nodelist=`hostname`
# We are using 1 task on this node and 5 CPUs (Threads). Have the dashboard listen to 0.0.0.0 to bind it to all
# network interfaces. This allows to access the dashboard through port-forwarding:
# Let's say the hostname=cluster-node-500 To view the dashboard on localhost:8265, set up an ssh-tunnel like this: (assuming the firewall allows it)
# $  ssh -N -f -L 8265:cluster-node-500:8265 user@big-cluster
srun --nodes=1 --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --nodelist=`hostname` ray start --head --block --dashboard-host 0.0.0.0 --port=6379 --num-cpus ${SLURM_CPUS_PER_TASK} &
sleep 5
# Make sure the head successfully starts before any worker does, otherwise
# the worker will not be able to connect to redis. In case of longer delay,
# adjust the sleeptime above to ensure proper order.

# Now we execute worker_num worker nodes on all nodes in the allocation except hostname by
# specifying --nodes=${worker_num} and --exclude=`hostname`. Use 1 task per node, so worker_num tasks in total
# (--ntasks=${worker_num}) and 5 CPUs per task (--cps-per-task=${SLURM_CPUS_PER_TASK}).
srun --nodes=${worker_num} --ntasks=${worker_num} --cpus-per-task=${SLURM_CPUS_PER_TASK} --exclude=`hostname` ray start --address $ip_head --block --num-cpus ${SLURM_CPUS_PER_TASK} &
sleep 5

python -u trainer.py ${total_cores} # Pass the total number of allocated CPUs
```

This script allocates 4 tasks, each on an individual cluster node. On the master node, think mpi rank 0, we execute the
ray head node and launch the dashboard. We bind to this rank with srun's --nodelist option.
The ray workers are running on all other nodes. It takes some time for ray to initialize, so sleep 5 seconds after
starting both the head node and the workers. And once the cluster is up and running, we can execute jobs on the
ray cluster. 

In the slurm file above we run trainer.py, a simple test script.

```
# trainer.py
from collections import Counter
import os
import sys
import time
import ray

num_cpus = int(sys.argv[1])

ray.init(address=os.environ["ip_head"])

print("Nodes in the Ray cluster:")
print(ray.nodes())

@ray.remote
def f():
    time.sleep(1)
    return ray.services.get_node_ip_address()

# The following takes one second (assuming that ray was able to access all of the allocated nodes).
for i in range(60):
    start = time.time()
    ip_addresses = ray.get([f.remote() for _ in range(num_cpus)])
    print(Counter(ip_addresses))
    end = time.time()
    print(end - start)
```

This script first prints out information of the ray nodes in the cluster. Then it executes `f` on the ray
worker nodes. In the for loop, each execution should take one second, since num_cpus is just the number
of total cores available. 



## Setting up a simple hyperparameter scan with Bayesian optimization
So, let's run ray and optimize hyperparameters for a neural network. Here I'm considering a
simple graph convolutional neural network where I can change 
* The size of an convolutional layer `conv_dim`
* The learning rate `lr`
* The batch size of the training data `batch_size`
* A weight decay parameter for my optimizer `weight_decay`


The code example below shows all the points I had to modify to get ray working with my training cycle.
In the main function, the first thing I set up are spaces for my hyperparameter. Here I follow
the [Configspace documentation](https://automl.github.io/ConfigSpace/master/). In prticular,
I'm sampling both `learning rate` and `weight_decay` logarithmically. The other parameters
`batch_size` and `conv_dim` can assume only discrete values.

The algorithm object `algo` is instantiated and instructed to minimize the variable
`mean_loss`. This variable has to be the one reported in the training loop inside the
function `train_network`.



https://docs.ray.io/en/releases-0.8.6/tune/api_docs/suggestion.html#bohb-tune-suggest-bohb-tunebohb

```python
import ray 
from ray import tune
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB

import ConfigSpace as CS

...

def train_network(config):
   """Trains my neural network and calculates the loss on my validation set"""

    model = my_neural_network(9, 3, config["conv_dim"]).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config["lr"],
                                 weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=3,
                                                           min_lr=1e-5)

    for i in range(num_epochs):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train(loader_train, optimizer, model)
        loss_dev = validate(loader_dev, model)
        scheduler.step(loss_dev)

        tune.report(mean_loss = loss_dev)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--ray-address",
        help="Address of Ray cluster for seamless distributed execution.")
    args = parser.parse_args()

    assert(args.ray_address)

    ray_head = args.ray_address + ":6379"
    ray.init(address=ray_head)

    # Define a search space for bayesian optimization
    # See https://docs.ray.io/en/releases-0.8.6/tune/api_docs/suggestion.html#bohb-tune-suggest-bohb-tunebohb
    config_space = CS.ConfigurationSpace()
    config_space.add_hyperparameter(CS.UniformFloatHyperparameter('lr', lower=1e-4, upper=1e-2, log=True))
    config_space.add_hyperparameter(CS.UniformFloatHyperparameter('weight_decay', lower=1e-5, upper=1e-2, log=True))
    config_space.add_hyperparameter(CS.CategoricalHyperparameter(name='batch_size', choices=[16, 32, 64, 128]))
    config_space.add_hyperparameter(CS.CategoricalHyperparameter(name='conv_dim', choices=[256, 512, 1024]))

    algo = TuneBOHB(config_space, max_concurrent=4, metric='mean_loss', mode='min')
    bohb = HyperBandForBOHB(time_attr='training_iteration',
                            metric='mean_loss',
                            mode='min',
                            max_t=100)
    analysis = tune.run(train_cgconv, name="train_bayes_v1", scheduler=bohb, search_alg=algo,
                        resources_per_trial={"gpu": 1, "cpu": 2}, num_samples=100)

```

To run hyperparameter optimization, submit the slurm script and wait until the job is executed.
After that, the results of the scan can be accessed as a dataframe:
```
from ray import tune
import pandas as pd

# Load results from the experiment
exp = tune.ExperimentAnalysis("/home/username/ray_results/train_bayes_v1")
# Access the experiment as a pandas dataframe
exp_df = exp.dataframe()
```

By default, ray stores the results of a run in `$HOME/ray_results` using subfolder names
corresponding to the `name` parameter that was passed in `tune.run`.


I hope reading this helps you getting started with hyperparameter tuning libraries.

