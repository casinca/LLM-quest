# Multimodal Qwen 3.5 from scratch


https://qwen.ai/blog?id=qwen3.5

If we could summarize the Qwen 3.5 architecture in a single sentence: *It's a Multimodal Qwen3-Next.*

They re-used the same ViT from Qwen3-VL and coupled it with their Qwen3-Next text model.

I'm not going to re-introduce the main text model, Qwen3-Next, it has been implemented from scratch and detailed
here: https://github.com/casinca/LLM-quest/blob/master/llm_quest/qwen/qwen3_next/


&nbsp;

## Changes from Qwen3-Next

These changes are from the Qwen3-Next implementation, and only concern text (later section for MRoPE):

- Switch from MoE to dense arch/FFN for loading smaller versions of the Qwen3.5 lineup and for local generation tests
- Fuse:
    - GatedAttention's Q and the Gate projections to match HF for loading pretrained weights: `MRoPEGatedAttention`
    - Q,K,V and the 3x 1D convolutional layers in GatedDeltaNet: `FusedGatedDeltaNet`
- More rigorous with dtypes now, all weights are in bf16 except:
    - `ZeroCenteredRMSNorm` in fp32 only for GDN (linear attention)
    - `log_A` in fp32 in GDN

&nbsp;

## Vision-specific changes

Compared to our original ViT `ViTModel` implemented in
[vision_transformer](https://github.com/casinca/LLM-quest/blob/master/llm_quest/multimodal/vision_transformer/README.md),
changes are architectural:

- We switch from 2D ($H \times W$ image only) to 3D patch embeddings (with extra temporal/time dimension, 
$T \times H \times W$)
- Q, K, V in vision attention (i.e., bidirectional/full/non-causal) are fused to load pretrained weights
- Q, K, V have biases added
- Specific Axial 2D RoPE (but og/classic positional is also kept on top)
- The Qwen3 ViT patch embedding also performs temporal downsampling (reducing the number of frames/time dimension)
  controlled by the `temporal_patch_size` argument, whereas our original `ViTAdapter` was only doing feature
  reprojection to the text model/LLM.
- We had a classification head for the ViT and the adapter was in the training loop.  
Here the adapter is integrated as
  the head of the ViT. i.e., the Qwen3 ViT output is directly reprojected to the embedding dimension of the text
  model/LLM.
- The head `ViTMergeAdapter` isn't just reprojecting, it also performs spatial downsampling controlled by the
  `spatial_merge_size` argument (reducing the number of patches, in both directions width and height, within an image).

&nbsp;

## Changes for the final Multimodal Qwen3.5 VLM


MRoPE (Multimodal RoPE/ 3D RoPE) takes over the classic 1D RoPE (Qwen3-Next) in order to make the model understand
spatio-temporal (T, H, W) positional information for our image/video patches.


### Details on Multimodal RoPE:

&nbsp;

### Multimodal Generation tests

In `qwen3_5_generate_multimodal.py`, we test the vision capabilities of the model with the same image (from flickr8k
dataset) fed to the from-scratch Multimodal GPT-2 implemented
[here](https://github.com/casinca/LLM-quest/blob/master/llm_quest/multimodal/README.md).

<img src="../../multimodal/_vlm_img/_vlm_img_test_sample.png" alt="alt text" width="300"/>

&nbsp;

The GPT-2 VLM was a 5-mins training job, so it's not a very fair comparison, the difference is obviously quite high:

```python
prompt = "What do you see in the image?"

max_gen=50, temp=1.0, seed=123

Our Qwen3.5-0.8B:

"<think>

</think>

In the image, two large Doberman pinschers are playing energetically in a snowy field. The animals are bickering or fighting — one is aggressively lying on its side while the other has its head thrown back, mouth open as if neighing"
```