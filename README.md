
# FlexNet Simulator 

This module contains the FlexNet simulator, which implements the co-optimization algorithm for DNN parallelization strategy search and topology construction. The source code was modified from the FlexFlow projcect. To check the original FlexFlow README file, please click [here](FF_README.md).

## Compilation
FlexNet follows the same build steps as the original FlexFlow project. Please check the [installation document](INSTALL.md) for how to compile and install the program. 

## Directory structure 
This section describes the relavent directory structure for FlexNet.
| Directory | Description |
|-----------|-------------|
| `config`    | Configuration files to generate the cmake build directory |
| `examples` | Example DNN implementations. `examples/cpp/*sim` are the DNNs used for the plots in the paper |
| `fbuf2` | FlatBuffer directory used for generating and storing taskgraph from FlexNet |
| `include` | Header files for the project |
| `src` | Soruce file of FlexNet simulator| 

## Usage
Please read the original [FlexFlow README](FF_README.md) first before proceeding.
To use the FlexNet simulator, there are a few essential steps. Here we will use the DLRM (`examples/cpp/DLRMsim`) as a running example. 

### Parameters
In addition to the parameters from original FlexFlow, FlexNet requires the following additional parameters to be specified:

`--interface-bandwidth <bw>`: bandwidth of each interface in Gbps.

`--network-latency <lat>`: Network link latency in microseconds.

`--degree <n>`: number of interface for each server. For Fat-Tree topology this has to be 1.

`--net-opt <0|1>`: whether the program should enable the TotientPerm algorithm. 1 for ture, 0 for false.

`--nsimnode <n>`: number of nodes to be simulated.

`--big-gpu <n>`: to enable emulating multiple GPU in one machine.

`--measure`: specify the run to be a measurement run, which generate the profiled data in a json file.

`--mfile <filename>`: when the run is not a measurement run, this flag specifies the path the profiled json file.

`--taskgraph <filename>`: output taskgraph of the run. The taskgraph stores the entire DNN training task DAG and is used for the FlexNetPacket simulator.

`--topology <topoopt|fattree|random>`: specify which network topology to simulate and search in this run. 


### Profiling 
FlexNet stores the profiled data for each DNN in a file, so we don't have to re-profile each DNN each time we run a simulation. For the p
Each measurement needs to specify the model, the cluster size, and the global batch size. For DLRM, for instance:
cd build/examples/cpp/DLRM_sim

`./dlrmsim -ll:gpu 1 -ll:cpu 1 -ll:zsize 20000 -ll:fsize 39000 -ll:util 4  -dm:memoize --embedding-bag-size 100 --arch-sparse-feature-size 256 --arch-embedding-size 10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000 --arch-mlp-bot 1024-1024-1024-1024 --arch-mlp-top 2048-2048-2048-2048-2048-2048-2048-2048-1 --search-budget 1 --interface-bandwidth 100 --inter-gpu-bandwidth 256 --gpu-dram-bandwidth 200 --network-latency 1 --net-opt 1 ---batch-size 4096  --nsimnode 16 --big-gpu 4 --simulator-workspace-size 38654705664 --measure`

After this, we will get a measure_<>_<>.json file that profiles all different ways of chopping this DNN. 

### Execution
With the measurement file, we run the MCMC and topology search with

`./dlrmsim -ll:gpu 1 -ll:cpu 1 -ll:zsize 20000 -ll:fsize 39000 -ll:util 4  -dm:memoize --embedding-bag-size 100 --arch-sparse-feature-size 256 --arch-embedding-size 10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000 --arch-mlp-bot 1024-1024-1024-1024 --arch-mlp-top 2048-2048-2048-2048-2048-2048-2048-2048-1 --search-budget 5000 --interface-bandwidth 100 --degree 4 --inter-gpu-bandwidth 256 --gpu-dram-bandwidth 200 --network-latency 1 --net-opt 1 --enable-propagation ---batch-size 4096  --nsimnode 16 --big-gpu 4 --simulator-workspace-size 65536 –mfile <measurement json from last step> --taskgraph <output taskgraph file name> --topology topoopt`

Note that the DNN model parameters, batch size and number of nodes in the cluster has to be the same as the measurement file.

### Output
After the last commnad, we should get the taskgraph as a flatbuffer file. This is what we will then feed into the packet simulator. Please see the README file of FlexNetPacket for details on how to run the packet simulation.


## Acknolwdgement
The author would like to thank Zhihao Jia, the original author of FlexFlow, for providing invaluable help and guidence on making TopoOpt happening. 