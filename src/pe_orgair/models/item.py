from pydantic import BaseModel
from typing import List
class Item(BaseModel):
    id: int
    name: str
    price: float
    description: str = None