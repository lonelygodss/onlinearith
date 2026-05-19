# Baseline MXFP Structured Sparsification

This repository provides an $N:M$ structured baseline sparsify metric compatible precisely with `MXFP` quantization workflows. The sparsification runs solely via offline metric calibration and masks out parameters directly prior to runtime. This ensures that inference involves completely native speed “standard calculations.”

## Mathematical Evaluation Metric
Consider a standard linguistic prediction projection task via parameterized continuous linear layers. The output corresponds directly to $Y = X \cdot W^T$.

For language models, the layer activation $X$ generally propagates a structured metric shaped $(N \times L, C_{in})$, where $N$ translates into batch dimensionality, and $L$ binds parallel sequence lengths.

In this technique, individual parameters are precisely ranked by evaluating metric scoring based directly off calibration inputs. To compute $X$ stability across diverse inputs, first the $L_2$ norm evaluates per-channel activation accumulation:
$$ \lVert X_j \rVert_2 = \sqrt{\sum_{k=1}^{N \times L} X_{kj}^2} $$

The calibration magnitude score pairs $W$ precisely against this static input channel norm:
$$S_{ij} = |W_{ij}| \cdot \lVert X_j \rVert_2$$

> **Note**: To ensure exact parity against `MXFP` baseline structures, $W_{ij}$ gets block-quantized explicitly to its local `MX` low-precision envelope mapping precisely *before* evaluating $|W_{ij}|$. This accounts natively for format truncation distortions upfront.

## Structured Masking Process
The resulting row matrix $S$ (shape $C_{out}, C_{in}$) reflects local significance.
Instead of globally scaling unconstrained unstructured parameters ($S \%$ pruning on standard sorted boundaries), this baseline methodology incorporates grouped masking parameter constraints natively into blocks defined by $M$:
- Across every row mapping projection, sequential channels form a structural boundary of $M$ group size.
- A local sort maps parameters strictly against this isolated dimension.
- The minimum threshold sequence composed strictly of length $N$ becomes permanently eliminated (its weights physically zeroed out).

## Workflows

We provide unified multi-GPU native scaling exactly identically simulating our internal structures. See the master `README` natively to launch directly on your local configurations.
