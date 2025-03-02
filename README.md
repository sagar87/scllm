# scllm

A Python package for annotating single-cell RNA sequencing data using Large Language Models.

[![PyPI version](https://badge.fury.io/py/scllm.svg)](https://badge.fury.io/py/scllm)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://haraldvohringer.com/scllm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

scllm leverages the power of Large Language Models to automatically annotate cell types in single-cell RNA sequencing data. It integrates seamlessly with scanpy and provides an intuitive interface for cell type annotation based on marker gene expression.

## Installation

You can install scllm using pip:

```bash
pip install scllm
```

Or using uv:

```bash
uv pip install scllm
```

## Quick Start

```python
import scanpy as sc
import scllm

# Load your data
adata = sc.read_h5ad('your_data.h5ad')

# Perform clustering if not already done
sc.tl.leiden(adata)

# Initialize your LLM (example with OpenAI)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI()

# Annotate clusters
adata = scllm.annotate_cluster(llm, adata, cluster_key='leiden')

# Access annotations
print(adata.obs['leiden_annotated'])
```

## Features

- Automatic cell type annotation using LLMs
- Seamless integration with scanpy
- Support for multiple LLM providers
- Interactive Jupyter notebook examples
- Customizable annotation parameters

## Documentation

For detailed documentation and examples, visit our [documentation page](https://haraldvohringer.com/scllm/).

Check out our example notebooks:

- [Cell Type Annotation](https://haraldvohringer.com/scllm/notebooks/cell_types.html)

## Requirements

- Python ≥ 3.10
- scanpy ≥ 1.11.0
- langchain ≥ 0.3.7
- And other dependencies listed in pyproject.toml

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use scllm in your research, please cite:

```bibtex
@software{vohringer2024scllm,
  author = {Vöhringer, Harald},
  title = {scllm: Single-Cell Annotation with Large Language Models},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/sagar87/scllm}
}
```

## Contact

Harald Vöhringer - [harald.voeh@gmail.com](mailto:harald.voeh@gmail.com)

Project Link: [https://github.com/sagar87/scllm](https://github.com/sagar87/scllm)
