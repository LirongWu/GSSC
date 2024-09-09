# Graph Structure Self-Contrasting (GSSC)


This is a PyTorch implementation of Graph Structure Self-Contrasting (GSSC), and the code includes the following modules:

* Dataset Loader (Cora, Citeseer, Actor, Coauthor-CS, and Coauthor-Phy)

* Various baselines (GCN, GraphSage, GAT, SGC, APPNP, and DAGNN compared in this paper)

* Neighborhood sparsification (NeighSparse) and Neighborhood self-contrasting (NeighContrast) network

* Bi-level optimization framework

* Training and evaluation paradigm 

  

## Main Requirements

* networkx==2.5
* numpy==1.19.2
* dgl==0.6.1
* torch==1.6.0



## Description

* main.py  
  * main() -- Train the model for node classification task with NeighContrast network on five datasets.
* model.py  
  * MLP_GNN() -- A pure MLP-based architecture ans two prediction heads.
  * EdgeSampler() -- Neighborhood Sparsification (NeighSparse) Network.
  * MixupScale() -- Learning Sampling Coefficients for interpolation between the target node and its ineighborhood nodes.
* dataset.py  

  * dataloader() -- Load five datasets as well as their variants with different label rates, label noise, and struture disturbance.
* utils.py  
  * SetSeed() -- Set seeds for reproducible results.



## Running the code

1. Install the required dependency packages

3. To get the results on a specific *dataset*, please run with proper hyperparameters:

  ```
python main.py --dataset data_name
  ```

where the *data_name* is one of the 5 datasets (Cora, Citeseer, Actor, Coauthor-CS, and Coauthor-Phy) . Use  *Cora* dataset an example: 

```
python main.py --dataset cora
```

3. To get the results on datasets with different label rates, label noise, and struture disturbance, please run with:

  ```
python main.py --label_mode
  ```

where the *label_mode* denotes different dataset variants. **(1) -1:** training with 1 labels per class; **(2) -3:** training with 3 labels per class; (3) -5: training with 5 labels per class; **(4) -10:** training with 5 labels per class; **(5) -25:** training with structure perturbation ratio 5%; **(6) -30:** training with structure perturbation ratio 10%; **(7) -40:** training with structure perturbation ratio 20%; **(8) -50:** training with structure perturbation ratio 30%; **(9) -51:** training with symmetric label noise ratio 20%; **(10) -52:** training with symmetric label noise ratio 40%; **(11) -53:** training with symmetric label noise ratio 60%; **(12) -54:** training with asymmetric label noise ratio 20%; **(13) -55:** training with asymmetric label noise ratio 40%; **(14) -56:** training with asymmetric label noise ratio 60%.



## License

Graph Structure Self-Contrasting (GSSC) is released under the MIT license.