# Learning Group Importance using the Differentiable Hypergeometric Distribution
Python library for the differentiable hypergeometric distribution.

This is the official code for the ICLR 2023 Paper (Spotlight) 
**"Learning Group Importance using the Differentiable Hypergeometric Distribution"**.

[Link to Openreview](https://openreview.net/forum?id=75O7S_L4oY&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2023%2FConference%2FAuthors%23your-submissions))

[Link to Arxiv](https://arxiv.org/abs/2203.01629)

We are still working on the code and the repository. Feedback and requests are very welcome.

## How to get started
We provide an environment file [```env_mvhg.yml```](env_mvhg.yml) that helps you with setting up a conda environment.
For help on how to install conda, please follow the guidelines on the offical webiste ([link to the offical website](https://anaconda.org/))

To create the conda environment needed, please run the following command

```bash
conda env create -f env_mvhg.yml
conda activate mvhg
pip install "[.pt]"
```
The conda environment runs on python 3.8.

## Minimal Example
We provide a minimal example, which learn the class weights from samples.
The minimal example uses pytorch lightning, weights & biases, and hydra config.

In the root directory, run the following command
```bash
python main_minimal_app.py
```

## Citation
If you use our model in your own, please cite us using the following citation
```
@inproceedings{sutter2023,
  title={Learning Group Importance using the Differentiable Hypergeometric Distribution},
  author={Sutter, Thomas M and Manduchi, Laura and Ryser, Alain and Vogt, Julia E},
  year = {2023},
  booktitle = {International Conference on Learning Representations},
}
```

## Questions, Requests, Feedback
For any questions or requests, please reach out to:
[Thomas Sutter](https://thomassutter.github.io/) [(thomas.sutter@inf.ethz.ch)](mailto:thomas.sutter@inf.ethz.ch)
