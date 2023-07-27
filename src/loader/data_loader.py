import abc
from typing import List

from src.loader.model_data import ModelData


class DataLoader(abc.ABC):
    @abc.abstractmethod
    def load(self) -> List[ModelData]:
        raise NotImplementedError(
            "load() not implemented for DataLoader"
        )
