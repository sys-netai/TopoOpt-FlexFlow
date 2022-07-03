This repository contains the full simulation code of TopoOpt. It contains two main part:
1. The FlexFlow-fork that does the MCMC search and topology construction. The relavent code located under src/runtime. 
3. The network packet simulator that performs detailed network simulation. The relavent code is located under the submodule "ffsim-opera" (https://github.com/Flasew/ffsim-opera).

Most simulations are run on MIT supercloud which utilizes the Slurm scheduler. A few sample scripts for running the DLRM experiment in the paper can be found under the "scripts" directory under the repository root. 

The testbed version of FlexFlow is NOT located under this repository. The code can be found at https://github.com/hipersys-team/topoopt_ff_testbed
The FlexFlow testbed code can be used to produce the resnet, vgg, transformer and candle result in the paper.

The testbed also requires the hacked version of NCCL found at https://github.com/hipersys-team/mccl, and RDMA forwarding setup. For the testbed setup, a detailed documentation on how to setup the RDMA forwarding can be found at https://docs.google.com/document/d/190nelkTXo7fEQNWRe4rnMglzAvV1jj-ZyShMcAGZH08/edit?usp=sharing. This document also contains how to setup the hacked version of nccl 

Two other testbed repositories exist. The first one is the pytorch-based TopoOpt testbed, which is located at https://github.com/hipersys-team/topoopt_testbed. This version is used to generate the testbed plot for ResNet time-to-accuracy plot for the NSDI 2022 submission.

The second one is the Facebook DLRM implementation, located at https://github.com/hipersys-team/dlrm. This codebase is used to generate the DLRM sensitivity analysis in the NSDI 2023 version of the paper. The file https://github.com/hipersys-team/dlrm/blob/main/run_a100.sh contains a detailed set of parameters for the 12-node testbed in the hipersys group. 



