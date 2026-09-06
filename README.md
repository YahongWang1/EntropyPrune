<div align="center">
  <h1 style="display: inline-block; margin: 0;">🔬 EntropyPrune: Matrix Entropy Guided Visual Token Pruning for Multimodal Large Language Models</h1>
</div>

<h4 align="center"> 

[Yahong Wang](https://scholar.google.com/citations?user=ps7AntYAAAAJ&hl=en)<sup>1</sup>,
[Juncheng Wu](https://chtholly17.github.io/)<sup>2</sup>,
[Zhangkai Ni](https://eezkni.github.io/)<sup>1✉</sup>,
Chengmei Yang<sup>1</sup>, 
[Yihang Liu](https://scholar.google.com/citations?user=Qsl7mMgAAAAJ&hl=zh-CN)<sup>1</sup>,<br>
Longzhen Yang<sup>1</sup>,
[Yuyin Zhou](https://yuyinzhou.github.io/)<sup>2</sup>,
Ying Wen<sup>3</sup>, 
Lianghua He<sup>1,4✉</sup>, 



<sup>1</sup>Tongji University, <sup>2</sup>University of California, Santa Cruz,<br>
<sup>3</sup>East China Normal University, <sup>4</sup>Shanghai Eye Disease Prevention and Treatment Center

</h4>

<div align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2602.17196-AD1C18.svg?logo=arXiv)](https://arxiv.org/abs/2602.17196)
</div>

## 📢 News

- **`2026.09.07`** Our [paper](https://arxiv.org/abs/2602.17196) and [Code](https://github.com/YahongWang1/EntropyPrune) are available!
- **`2026.08.21`** Our paper is accepted at EMNLP 2026 Findings!

Star 🌟 us if you think it is helpful!!

## ⚡Introduction
<p align='center'>
<img src='https://github.com/YahongWang1/EntropyPrune/blob/main/images/overview.png' alt='mask' width='1000px'>
</p>

> **TLDR:** By analyzing the layer-wise matrix entropy of visual representations, we uncover an “Entropy Collapse Layer” (ECL) where visual-token information drops sharply, providing a principled answer to when to prune instead of relying on manually chosen layers. Building on this, EntropyPrune measures each token’s information content with token-wise matrix entropy to determine what to prune, while a dual-Gram-matrix spectral formulation makes entropy computation efficient.

## 🛠 Preparation
### LLaVA
1. Clone this repository.

```bash
git clone https://github.com/YahongWang1/EntropyPrune
cd EntropyPrune
```

2. Environment Setup.

```Shell
 conda create -n entropy python=3.10 -y
 conda activate entropy
 pip install -e .
 pip install flash_attn==2.5.9.post1 --no-build-isolation
```

3. Download Benchmark.

Please follow instructions in [LLaVA-Evaluation](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).

### VLMEvalKit

VLMEvalKit uses separate environments for LLaVA and Qwen2.5-VL because they require different dependency versions.

#### LLaVA Environment

```Shell
# Create the Python 3.10 environment for LLaVA-1.5 evaluation.
conda create -n vlmeval-llava15 python=3.10 -y
conda activate vlmeval-llava15

# Install the exported dependencies and VLMEvalKit without resolving them again.
python -m pip install -r VLMEvalKit_EntropyPrune/requirements/vlmeval-llava15.txt
python -m pip install -e ./VLMEvalKit_EntropyPrune --no-deps
```

#### Qwen2.5-VL Environment

```Shell
# Create the Python 3.10 environment for Qwen2.5-VL evaluation.
conda create -n vlmeval-qwen25vl python=3.10 -y
conda activate vlmeval-qwen25vl

# Install FlashAttention after PyTorch, then install VLMEvalKit without changing dependencies.
python -m pip install -r VLMEvalKit_EntropyPrune/requirements/vlmeval-qwen25vl.txt
python -m pip install flash-attn==2.7.3 --no-build-isolation
python -m pip install -e ./VLMEvalKit_EntropyPrune --no-deps
```



## 🎯 Usage
### LLaVA-1.5-7B (Table 1)
```Shell
# Results of MME ScienceQA TextVQA 
conda activate entropy
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/5bench.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/flops.sh # Calculate FLOPs
```

```Shell
# Using VLMEvalKit for more benchmarks
cd VLMEvalKit_EntropyPrune
conda activate vlmeval-llava15
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/llava15.sh
```

### LLaVA-1.6-7B (Table 2)
```Shell
# Results of TextVQA MM-Vet
conda activate entropy
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_6/eval/5bench.sh
```

```Shell
# Using VLMEvalKit for more benchmarks
cd VLMEvalKit_EntropyPrune
conda activate vlmeval-llava15
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_6/llava16.sh
```

### Qwen-2.5-VL-7B (Table 3)
```Shell
# Using VLMEvalKit
cd VLMEvalKit_EntropyPrune
conda activate vlmeval-qwen25vl
CUDA_VISIBLE_DEVICES=0 bash scripts/qwen25vl/qwen25vl.sh
```




## 📌 Citation

```bibtex
@misc{wang2026entropyprunematrixentropyguided,
      title={EntropyPrune: Matrix Entropy Guided Visual Token Pruning for Multimodal Large Language Models}, 
      author={Yahong Wang and Juncheng Wu and Zhangkai Ni and Chengmei Yang and Yihang Liu and Longzhen Yang and Yuyin Zhou and Ying Wen and Lianghua He},
      year={2026},
      eprint={2602.17196},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.17196}, 
}
```


## ❤️ Acknowledgment
Thanks to the open-source contributions of [LLaVA](https://github.com/haotian-liu/LLaVA), [DART](https://github.com/ZichenWen1/DART), and [VLMEvalKit](https://github.com/open-compass/vlmevalkit).



## 📭 Contact
For any questions about our paper or code, please email `yahongwang@tongji.edu.cn`.
