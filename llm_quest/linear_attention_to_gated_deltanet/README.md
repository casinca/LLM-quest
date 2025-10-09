# From Linear Attention to Gated DeltaNet

*These are only notes to better grasp the genesis of the subquadratic linear attention. Its evolution up to the current
SOTA Gated DeltaNet architecture that is used in the [Qwen3-Next
implementation](https://github.com/casinca/LLM-quest/tree/master/llm_quest/qwen/qwen3_next).*  

*Credit to [@sustcsonglin](https://github.com/sustcsonglin) for her clear and detailed
[blogpost](https://sustcsonglin.github.io/blog/2024/deltanet-1/) which these simpler notes are inspired from.*

&nbsp;

### Linear Attention

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

but here, just removing the softmax, nothing changes, we're still doing 
$\mathbf{Q}\mathbf{K}^\mathsf{T} \in\mathbb{R}^{L \times L}$.  
But by removing the softmax and now thanks to associativity, we are not locked anymore by this order of operations,
therefore we can do:

$$\mathbf{O} = \mathbf{Q}(\mathbf{K}^\mathsf{T}\mathbf{V})$$

&nbsp;

*Why this re-arrangement matters and how does it make linear attention subquadratic?*  

As we mentioned, $\mathbf{Q}, \mathbf{K}, \mathbf{V}, \in \mathbb{R}^{L \times d}$, so the matmul
$\mathbf{K}^\mathsf{T}\mathbf{V}\in \mathbb{R}^{d \times d}$ which is where the nice time complexity improvement comes
from.   
Yes nothing changes afterward, if we call $\mathbf{S} = \mathbf{K}^\mathsf{T}\mathbf{V}$, then $\mathbf{Q}\mathbf{S}$
is still $\in \mathbb{R}^{L \times d}$.

But overall the Attention complexity was reducing from $\mathcal{O}(L^2d)$ to $\mathcal{O}(Ld^2)$, **Linear attention is
born.**

As soon as $L$ (grows with input) > $d$ (which is a fixed hparam) Linear attention becomes worth it.  



TODO too vague for inference need to mention commutativity and distributivity below

&nbsp;

Now for inference same principle, with more granularity, for a given time step $t$, and query, key, value vectors 
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
$\mathbf{S}_{t-1}$ is the very foundation of linear attention.  
Starting from now on, it's easier to understand the subsequent variants cited here because they are **all** variations
about how to compute and update this state matrix $\mathbf{S}_t$ and also aim to mitigate drawbacks of the original
linear attention.

### Drawbacks

The small $d \times d$ state matrix is nice for speed but as Songlin mentions, since it can't grow, perfectly keeping
all the details (compression) from all previous informations is challenging and a main point of underperformance
compared to classic attention.

As we can see in the boxed formula, the state $S_t$ at each timestep is formed by adding the new outer product 
$\mathbf{v}_t \mathbf{k}_t^\mathsf{T}$ directly to the accumulated state of all previous steps $\mathbf{S}_{t-1}$.  
This accumulation of previous states without any regulation, (ie potentially accumulating unwanted
noisy or now irrelevant informations) creates **memory overload**. This overload is the root cause of retrieval
errors and poor performance

### Gated Linear variants

To reduce that problem and control how much of the previous state $\mathbf{S}_{t-1}$ we want to keep/remember for the
next state $\mathbf{S}_t$, they introduced a gating term, aka **forgetting mechanism**, to scale $\mathbf{S}_{t-1}$.

This gave birth to gated linear attention variants like [GLA](https://arxiv.org/abs/2312.06635),
[Mamba](https://arxiv.org/abs/2312.00752). For our Qwen3-Next case, if we omit delta rule for now, the gate is the alpha
term from GDN:

$$
\boxed{\mathbf{S}_t = \alpha_t \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\mathsf{T}}
$$

&nbsp;

### DeltaNet

Now onto how we can also optimize the outer product addition.  
For a given time step, instead of simply adding the outer product $\mathbf{v}_t \mathbf{k}_t^\mathsf{T}$ to the previous
state $\mathbf{S}_{t-1}$, researchers did a smart adaptation of the [delta
ruled](https://proceedings.mlr.press/v139/schlag21a.html) for this purpose (hence the name [DeltaNet](https://proceedings.mlr.press/v139/schlag21a.html)):

Instead of using the current value $\mathbf{v}_t$ directly for the outer product with $\mathbf{k}_t^\mathsf{T}$, they
instead use an error adjusted value, let's call it  $\Delta \mathbf{v}_t$.  
$\Delta \mathbf{v}_t$ is simply the difference between the current value $\mathbf{v}_t$ and the model's prediction of
this value (retrieved by querying the previous state $\mathbf{S}_{t-1}$ with the current key $\mathbf{k}_t$):

$$
\Delta \mathbf{v}_t = \mathbf{v}_t - \mathbf{S}_{t-1} \mathbf{k}_t
$$

> This is extremely similar to the residual error in classic ML (as in residual = target - prediction).  
>Here we could say we're using the value residual error, where $\mathbf{v}_t$ is the target and $\mathbf{S}_{t-1}
>\mathbf{k}_t$ is the prediction.

&nbsp;

so our outer product becomes $\Delta \mathbf{v}_t \mathbf{k}_t^\mathsf{T}$ and our $S_t$ (*omitting previous $\alpha$
gate for now*):

$$
\mathbf{S}_t = \mathbf{S}_{t-1} + \Delta \mathbf{v}_t \mathbf{k}_t^\mathsf{T}
$$

Furthermore, analogous to SGD/optimizer updates, this outer product $\Delta \mathbf{v}_t \mathbf{k}_t^\mathsf{T}$ is also
governed by a learning rate $\beta$ (*this is our beta from Qwen3-Next GDN impl*), in order to control how much of the
new information we want to add, so in the end we have:

$$
\boxed{\mathbf{S}_t = \mathbf{S}_{t-1} + \beta \Delta \mathbf{v}_t \mathbf{k}_t^\mathsf{T}}
$$

&nbsp;

Now we're getting closer to the final Gated DeltaNet, which as the name suggests, is the combination of both
optimizations we just saw, ie linear attention with a gate and using the delta rule.

### Gated DeltaNet

Combining both optimizations, 


TODO
TODO check github latex rendering!

&nbsp;

# Acknowledgments

TODO

- Gated Linear Attention: https://arxiv.org/abs/2312.06635
- Gated Attention: https://arxiv.org/abs/2505.06708
- Songlin Yang blogpost on DeltaNet: https://sustcsonglin.github.io/blog/2024/deltanet-1/
- Linear attention: https://arxiv.org/abs/2006.16236
- Gated Delta Networks: https://arxiv.org/abs/2412.06464
- DeltaNet: https://proceedings.mlr.press/v139/schlag21a.html
- Delta rule: https://direct.mit.edu/books/edited-volume/5431/chapter-abstract/3958517/1960-Bernard-Widrow-and-Marcian-E-Hoff-Adaptive
- Mamba: https://arxiv.org/abs/2312.00752
