# Vision Transformer (ViT) from scratch

This from-scratch ViT implementation is an intermediate step towards multimodal models. The goal is to reuse the ViT as
an image encoder for an LLM.

It is based on the Vision Transformer architecture depicted in the paper ["An Image is Worth 16x16 Words: Transformers
for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) by Dosovitskiy et al.  

> We are adapting our existing GPT decoder architecture to work as a transformer encoder

## Architectural Changes

### Input Processing: Tokens → Image Patches

With text, we tokenize a sequence and then process token embeddings + positional embeddings. Similarly, here we split an
image as if it was a sequence, into equal-sized patches that could be seen as words (*hence the paper's name*) and then
project them to `emb_dim`. The whole process is done efficiently via a convolutional layer in the `PatchEmbedding` class.

Concerning Positional Encoding, unlike in GPT, we don't need an `nn.Embedding` table because we are not dealing with
dynamic sequence lengths. In our case all images in a batch have the same `CxHxW` size and therefore the same number of
patches, thus we can use a single 3D tensor, with `nn.Parameter`, of shape `(1, num_patches + 1, emb_dim)`.

*Why the length of our sequence is `num_patches + 1`?*  
In order to get the model's prediction of the image, we prepend a learnable classification token/patch at the
beginning of the sequence, we will retrieve the hidden state of that patch as a proxy for the model's representation of
the image.   

*Why does just getting the first token work compared to a classic decoder architecture?* It's the consequence of our 2nd
change below:

### Removing Causal Masking

The GPT arch uses causal masks to prevent looking at future tokens (autoregressive). Here we are not trying to
predict the next patch, but to get the whole picture i.e. we do want all patches to attend to all other patches
simultaneously, hence we switch to full attention (`ViTMultiHeadAttention` class).

### Output: Next token prediction → Image Classification

We still need to reduce the hidden state's dimensions of the classification token in order to interpret it as a class
prediction. This is done by down-projecting the dimensions to the number of classes via a simple
linear layer (`self.classifier` in `ViTModel` class).

This is very similar to GPT, we're just slicing first to retrieve the class token and then project to the desired
dimensions:

```python
# GPT output
logits = self.out(x)  # (batch, seq_len, vocab_size)

# ViT output
cls_token_output = x[:, 0]  # classification token
logits = self.classifier(cls_token_output)  # (batch, num_classes)
```

## Towards Multimodal LLMs


TODO


Model Configs are in `config.py`.