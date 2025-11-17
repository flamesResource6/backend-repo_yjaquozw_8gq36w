from pydantic import BaseModel, Field
from typing import Optional

class Search(BaseModel):
    user_id: str = Field(..., description="Supabase user id")
    card_name: str = Field(..., description="Card name as asked by user")
    last_sold_price: Optional[float] = Field(None, description="Most recent sold price in GBP")
    last_sold_date: Optional[str] = Field(None, description="Date of most recent sale (ISO)")
    median: Optional[float] = Field(None, description="Median of last 10 UK sales in GBP")
    average: Optional[float] = Field(None, description="Average of last 10 UK sales in GBP")
