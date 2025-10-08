# From Linear Attention to Gated DeltaNet

*These are only notes to better grasp the genesis of the subquadratic linear attention. Its evolution up to the current
SOTA Gated DeltaNet architecture that is used in the [Qwen3-Next implementation](https://github.com/casinca/LLM-quest/tree/master/llm_quest/qwen/qwen3_next).*

&nbsp;

Starting from the beginning and the classic quadratic attention equation:  
(*omitting scaling and elem-wise mult masking for simplicity*)

$$
\mathbf{O} = \text{softmax}(\mathbf{Q}\mathbf{K}^\mathsf{T})\mathbf{V}
$$

all matrices are of shape (seq_len($L$), head_dim($d$)), ie 
$\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{O} \in \mathbb{R}^{L \times d}$

&nbsp;

In standard attention, we are locked in for the order of operations by the softmax term, we need to do $\mathbf{Q}\mathbf{K}^\mathsf{T}$ first
before the matmul with $\mathbf{V}$. On top of that, the raw attention scores $\mathbf{Q}\mathbf{K}^\mathsf{T} \in
\mathbb{R}^{L \times L}$ , which gives its known bottleneck quadratic nature.

[Linear attention](https://arxiv.org/abs/2006.16236) comes from removing the softmax and re-arranging the terms, made possible by the associative property
with matrices.  
So with linear attention, the equation becomes:
$$\mathbf{O} = (\mathbf{Q}\mathbf{K}^\mathsf{T})\mathbf{V}$$

but here, just removing the softmax, nothing changes, we're still doing $\mathbf{Q}\mathbf{K}^\mathsf{T}
\in\mathbb{R}^{L \times L}$.  
But by removing the softmax and now thanks to associativity, we are not locked anymore by this order of operations,
thefore we can do:

$$\mathbf{O} = \mathbf{Q}(\mathbf{K}^\mathsf{T}\mathbf{V})$$

&nbsp;

*why this re-arrangement matters and how does it make linear attention subquadratic?*  

As we mentioned, $\mathbf{Q}, \mathbf{K}, \mathbf{V}, \in \mathbb{R}^{L \times d}$, so the matmul
$\mathbf{K}^\mathsf{T}\mathbf{V}\in \mathbb{R}^{d \times d}$ which is where the nice time complexity improvement comes from.  
Yes nothing changes afterward, if we call $\mathbf{S} = \mathbf{K}^\mathsf{T}\mathbf{V}$, then $\mathbf{Q}\mathbf{S}$
is still $\in \mathbb{R}^{L \times d}$.

But overall the Attention complexity was reducing from $\mathcal{O}(L^2d)$ to $\mathcal{O}(Ld^2)$, **Linear attention is
born.**

TODO  cons vs classic attention
TODO too vague for inference need to mention commutativity and distributivity below

&nbsp;

Now for inference same principle, with more granularity, for a a givent time step $t$, and query, key, value vectors 
$\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t \in \mathbb{R}^{d}$, we can easily swap order however we want by leveraging
column or row
vectors, ie $\in \mathbb{R}^{1 \times d}$ or $\in \mathbb{R}^{d \times 1}$.

$$
\mathbf{o}_t = \sum_{j=1}^{t} (\mathbf{q}_t^\mathsf{T} \mathbf{k}_j)\mathbf{v}_j
$$
$$
= \sum_{j=1}^{t} \mathbf{v}_j(\mathbf{k}_j^\mathsf{T} \mathbf{q}_t)
$$
$$
= \left(\sum_{j=1}^{t} \mathbf{v}_j \mathbf{k}_j^\mathsf{T}\right) \mathbf{q}_t
$$

This is from this last re-arrangement and our outer product $\mathbf{v} \otimes \mathbf{k}^\mathsf{T}$ that they define the state matrix $\mathbf{S}_t = \sum_{j=1}^{t} \mathbf{v}_j
\mathbf{k}_j^\mathsf{T}$.  

$$
\mathbf{o}_t = \mathbf{S}_t \mathbf{q}_t
$$


By recurrence we can write $\mathbf{S}_t$ as:
$$
\boxed{\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\mathsf{T}}
$$

&nbsp;

This version by expressing state matrix $\mathbf{S}_t \in \mathbb{R}^{d \times d}$ based on its previous state 
$\mathbf{S}_{t-1}$
(the whole thing looks very Markovian btw) is the very foundation of linear attention.  
Starting from now on, it's easier to understand the subsequent variants studied here because they are **all** variations
about how to compute and update this state matrix $\mathbf{S}_t$.

TODO
TODO check github latex rendering!

# Acknowledgments

TODO




