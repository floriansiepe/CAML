# CAML

Source code for the paper "[A Few Models to Rule Them All: Aggregating Machine Learning Models](https://floriansiepe.de/publication/caml/Model_Clustering___LWDA_2023_Paper.pdf)" at LWDA 2023.

## Installation

```bash
poetry install
poetry shell
```

## Getting Started

See [demo.py](src/demo.py) for a simple example.

This implementation of CAML currently supports time series forecasting models only provided by
the [Darts](https://github.com/unit8co/darts) library. However, it can easily adapted for other libraries and model
types (e.g. sklearn).

### Adding model architectures

To add a new model architecture, you need to create a new class that inherits
from [`ObjectiveFactory`](src/aggregation/objective_factories/objective_factory.py) and implements the `create`
and `build_model` methods. See also the [prebuilt factories](src/aggregation/objective_factories) for examples.

## Citation

If this is useful for your research, please consider citing the paper ([PDF](https://floriansiepe.de/publication/caml/Model_Clustering___LWDA_2023_Paper.pdf)):

```bibtex
@inproceedings{Siepe2023,
    author = {Florian Siepe and Phillip Wenig and Thorsten Papenbrock},
    title = {A Few Models to Rule Them All: Aggregating Machine Learning Models},
    booktitle = {Proceedings of the Conference Lernen, Wissen, Daten, Analysen},
    numpages = {12},
    year = {2023},
    series = {LWDA '23},
    location = {Marburg, Germany},
    publisher = {CEUR Workshop Proceedings},
    url = {https://ceur-ws.org/XXXX}
}
```
