import time

# ring
epochs = "2"
dist_optimizer = "neighbor_allreduce"
seed = "50"
topology = "ring"
base_lr = "0.025"
momentum = "0.0" 
wd = "3e-4"

timearray=time.localtime(float(time.time()))
tt=time.strftime('%Y-%m-%d-%H-%M-%S',timearray)
filename="heavy_tail_epochs" + epochs + "_optimizer_" + dist_optimizer + "_topology_" + topology \
    + "_base_lr_" + base_lr + "_momentum_" + momentum + "_wd_" + wd + "_" + tt + ".log"

command = "BLUEFOG_OPS_ON_CPU=1 bfrun -np 8 python examples/decentralized_heavy_tail.py --epochs " + epochs + " --dist-optimizer " + dist_optimizer \
    + " --seed " + seed + " --atc-style --dirichlet-beta -1 --nu 1 --topology " + topology + " --base-lr " + base_lr + " --momentum " + momentum \
    + " --wd " + wd + " | tee " + filename + "\n"

# write command to shell files
f = open("./scripts/run_heavy_tail.sh", "w")
f.write(command)

# hypercube
topology = "hypercube"

filename="heavy_tail_epochs" + epochs + "_optimizer_" + dist_optimizer + "_topology_" + topology \
    + "_base_lr_" + base_lr + "_momentum_" + momentum + "_wd_" + wd + "_" + tt + ".log"

command = "BLUEFOG_OPS_ON_CPU=1 bfrun -np 8 python examples/decentralized_heavy_tail.py --epochs " + epochs + " --dist-optimizer " + dist_optimizer \
    + " --seed " + seed + " --atc-style --dirichlet-beta -1 --nu 1 --topology " + topology + " --base-lr " + base_lr + " --momentum " + momentum \
    + " --wd " + wd + " | tee " + filename + "\n"

f = open("./scripts/run_heavy_tail.sh", "a")
f.write(command)

# gradient-allreduce
dist_optimizer = "gradient_allreduce"

filename="heavy_tail_epochs" + epochs + "_optimizer_" + dist_optimizer \
    + "_base_lr_" + base_lr + "_momentum_" + momentum + "_wd_" + wd + "_" + tt + ".log"

command = "BLUEFOG_OPS_ON_CPU=1 bfrun -np 8 python examples/decentralized_heavy_tail.py --epochs " + epochs + " --dist-optimizer " + dist_optimizer \
    + " --seed " + seed + " --atc-style --dirichlet-beta -1 --nu 1 --base-lr " + base_lr + " --momentum " + momentum \
    + " --wd " + wd + " | tee " + filename + "\n"

f = open("./scripts/run_heavy_tail.sh", "a")
f.write(command)