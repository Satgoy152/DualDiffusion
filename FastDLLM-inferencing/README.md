## Usage of FastDLLM with Steps

The FastDLLM-7B folder is a copy of this [FastDLLM model](https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_7B) from HuggingFace (paper referenced below).

It contains everything except the weights and tokenizer, which are pulled in from HuggingFace to avoid storage issues.

## How to Use

All steps are laid out in the jupyter notebook, make sure the folder containing FastDLLM is in the same directory as where it is being used, or make sure to modify the path to reflect any changes.

Reference:

@misc{wu2025fastdllmv2efficientblockdiffusion,
      title={Fast-dLLM v2: Efficient Block-Diffusion LLM}, 
      author={Chengyue Wu and Hao Zhang and Shuchen Xue and Shizhe Diao and Yonggan Fu and Zhijian Liu and Pavlo Molchanov and Ping Luo and Song Han and Enze Xie},
      year={2025},
      eprint={2509.26328},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.26328}, 
}
