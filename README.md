<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About This Project

This repository contains code for the first unified open-source framework for temporal KG completion models  proposed in [A Re-evaluation of Temporal Knowledge Graph Completion Models under a Unified Framework](https://aclanthology.org/2021.emnlp-main.639.pdf). This framework provides full composability, where temporal embeddings, score functions, loss functions, regularizers, and the explicit modeling of reciprocal relations can be combined arbitrarily. You can be free from the nuisance, e.g. data processing, training configuring, hyperparameter tuning, and metrics evaluating.

Here's our motivation:
* All the experiments should be reproducible and comparable under a unified framework particularly for temporal knowledge graph embedding.
* Hyper-parameter searching and model tuning should be least pivotal when training new models.




<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

* python3 >= 3.7

* pytorch

* pyyaml

* numba

* arrow

  

<!-- USAGE EXAMPLES -->

## Usage

 ```
 # train a model
 python tkge.py train --config path-to-your-config-folder/config-default.yaml

 # eval a model
 python tkge.py eval --config path-to-your-config-folder/config-default.yaml

 # hyper-parameter optimization
 python tkge.py hpo --config path-to-your-config-folder/config-default.yaml

 # resume training
 python tkge.py resume --ex /path/to/folder
 ```




<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

Zhen Han - hanzhen02111@163.com; Gengyuan Zhang - gengyuanmax@gmail.com




<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [PyKEEN](https://github.com/pykeen/pykeen)
* [AllenNLP](https://github.com/allenai/allennlp)
* [LibKGE](https://github.com/uma-pi1/kge)
