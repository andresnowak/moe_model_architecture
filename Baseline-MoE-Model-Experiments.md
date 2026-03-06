# Baseline MoE Model Experiments

> **Megatron commit**: [https://github.com/swiss-ai/Megatron-LM/tree/merge-260109](https://github.com/swiss-ai/Megatron-LM/tree/merge-260109)
**Datasets**: "/iopsstor/scratch/cscs/anowak/datasets/megatron/llama_tokenized/fineweb-edu-100B/fineweb-edu-100B_00002_tokens”
**Image**: `/iopsstor/scratch/cscs/gfu/ce-images/megatron_deepep-aarch64.sqsh`
> 

## 03/03/2026 Draft model experiment

A draft model to make the experiment running and check:

- [x]  deepep
- model config
  
    ```json
    0: [DistributedDataParallel(
    0:   (module): Float16Module(
    0:     (module): GPTModel(
    0:       (embedding): LanguageModelEmbedding(
    0:         (word_embeddings): VocabParallelEmbedding()
    0:         (embedding_dropout): Dropout(p=0.0, inplace=False)
    0:       )
    0:       (rotary_pos_emb): RotaryEmbedding()
    0:       (decoder): TransformerBlock(
    0:         (layers): ModuleList(
    0:           (0-4): 5 x TransformerLayer(
    0:             (input_layernorm): IdentityOp()
    0:             (self_attention): SelfAttention(
    0:               (core_attention): TEDotProductAttention(
    0:                 (flash_attention): FlashAttention()
    0:                 (fused_attention): FusedAttention()
    0:                 (unfused_attention): UnfusedDotProductAttention(
    0:                   (scale_mask_softmax): FusedScaleMaskSoftmax()
    0:                   (attention_dropout): Dropout(p=0.0, inplace=False)
    0:                 )
    0:               )
    0:               (linear_proj): TERowParallelLinear(in_features=2048, out_features=2048, bias=False, TP=1)
    0:               (linear_qkv): TELayerNormColumnParallelLi
    0: near(in_features=2048, out_features=2560, bias=False, TP=1)
    0:               (q_layernorm): IdentityOp()
    0:               (k_layernorm): IdentityOp()
    0:             )
    0:             (pre_cross_attn_layernorm): IdentityOp()
    0:             (cross_attention): IdentityOp()
    0:             (cross_attn_bda): IdentityFuncOp()
    0:             (pre_mlp_layernorm): RMSNorm()
    0:             (mlp): MoELayer(
    0:               (router): TopKRouter()
    0:               (experts): TEGroupedMLP(
    0:                 (linear_fc1): TEColumnParallelGroupedLinear()
    0:                 (linear_fc2): TERowParallelGroupedLinear()
    0:               )
    0:               (shared_experts): SharedExpertMLP(
    0:                 (linear_fc1): TEColumnParallelLinear(in_features=2048, out_features=1536, bias=False, TP=1)
    0:                 (linear_fc2): TERowParallelLinear(in_features=768, out_features=2048, bias=False, TP=1)
    0:               )
    0:             )
    0:           )
    0:         )
    0:         (final_layernorm): RMSNorm()
    0:       )
    0:       (output_layer): ColumnParallelLinear(in_features=2048, out_features=13107
    0: 2, bias=False, TP=1)
    0:     )
    0:   )
    0: )]
    ```
    

## 05/03/2026 Reference Baseline MoE Model experiment

### Model

- **Qwen3-30B-A3B**
  
    ```json
    {
      "attention_bias": false,
      "attention_dropout": 0.0,
      **"head_dim": 128,**
      **"hidden_size": 2048,**
      "initializer_range": 0.02,
      **"intermediate_size": 6144,**
      "max_position_embeddings": 40960,
      **"moe_intermediate_size": 768,**
      "norm_topk_prob": true,
      **"num_hidden_layers": 48,**
      **"num_attention_heads": 32,
      "num_key_value_heads": 4,**
      **"num_experts": 128,**
      **"num_experts_per_tok": 8,
      "optimizer": "adam", # ademamix
      "act": "swiglu"**
    }
    ```
    
    - traditional GQA (4/32)
    - standard MoE sparsity (8/128)
    - reasonable model size

### Setup and Scripts

- **Megatron Recipe**: [link](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/qwen.html#id6)
- **Num of Node: 2**
- **Parallel Strategy and Estimated Memory per GPU**
    - PP4-EP2-DP2-TP1
    - GBS = 96
    - MBS = 4
    - GA = 12
    - SeqLen = 4096
    - Theoretical = 99150.76 MB (99.1GB)
        - Weight + optimizer = 44.18GB
        - Activation = 54.97GB
        - MoE activation recompute
        
        ```
        6: [Rank 6] (after 2 iterations) memory (MB) | allocated: 53199.095703125 | **max allocated: 84019.3681640625** | reserved: 84732.0 | max reserved: 84732.0
        2: [Rank 2] (after 2 iterations) memory (MB) | allocated: 50314.07666015625 | **max allocated: 81835.0830078125** | reserved: 82324.0 | max reserved: 82324.0
        4: [Rank 4] (after 2 iterations) memory (MB) | allocated: 50317.25390625 | **max allocated: 71306.03662109375** | reserved: 71744.0 | max reserved: 71744.0
        ```
    
- **Sbatch Script**: `/users/gfu/frameworks/Megatron-LM-sai/myscripts/apertus_qwen_30b_a3b_baseline.sh`

### Experiments

TBU