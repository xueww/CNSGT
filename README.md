# CNSGT: Generative Transformer for De Novo Drug Design Targeting the Central Nervous System

This repository contains the official source, datasets for the paper **"CNSGT: Generative Transformer for De Novo Drug Design Targeting the Central Nervous System"**.

## Overview

CNSGT is a novel Transformer-based generative framework that integrates a variational autoencoder (VAE) and property-conditioned attention mechanisms to address the unique challenges of Central Nervous System (CNS) drug design. The model is pre-trained on a large, CNS-focused molecular dataset and then fine-tuned for target-specific generation, as demonstrated in our case study on inhibitors for the Dopamine Transporter (DAT).

## Key Features

- **Property-Conditioned Generation**: Embeds 6 key CNS physicochemical properties (MW, LogP, LogD, TPSA, HBD, pKa) as conditions to guide the model toward generating molecules with ideal drug-like properties.
- **Transformer-based VAE Architecture**: Leverages the powerful sequence modeling capabilities of Transformers to capture long-range dependencies in SMILES strings and uses a VAE framework to ensure the novelty and diversity of generated molecules.
- **Transfer Learning for Target-Specific Design**: Enables the model to generate candidate compounds with high binding potential for a specific target (e.g., DAT) by fine-tuning on an elite set of molecules with high affinity for the target pocket.
- **End-to-End Validation Workflow**: The project covers not only the model itself but also a complete validation pipeline, from molecular docking and molecular dynamics (MD) simulations to synthetic route analysis by expert chemists.

## Workflow Overview

The computational workflow for this study is as follows:

1. **Data Preparation**: Curation and processing of over 500,000 CNS-like molecules from the ChemBridge-MPO database.
2. **Model Pre-training**: Training the CNSGT model on the large-scale dataset to learn the general rules of the CNS chemical space.
3. **Target-Specific Fine-tuning**:
   - Generating a large library of molecules using the pre-trained model.
   - Screening for elite molecules with high affinity and high MPO scores via molecular docking to 4 different conformations/sites of DAT.
   - Fine-tuning the pre-trained model separately on these four small, elite molecular sets.
4. **Molecule Generation and Validation**:
   - Generating new candidate molecules using the fine-tuned models.
   - Conducting in-depth validation of the best candidates through molecular docking, MD simulations, and synthetic route analysis.

## Citation

If you use CNSGT in your research, please cite our paper:

```bibtex
@article{cnsgt2025,
  title = {CNSGT: Generative Transformer for De Novo Drug Design Targeting the Central Nervous System},
  author={Yingjun Chen, Ding Luo, Shengneng Chen, Tingting Hou,‡ Chao Huang, and Weiwei Xue},
  journal={Journal of Chemical Information and Modeling},
  year={2025},
  volume={XX},
  pages={YYYY-ZZZZ
  doi = {10.xxxx/xxxxxx}
}
