# src/api/schemas.py

from pydantic import BaseModel, validator


class Customer(BaseModel):
    credit_score     : float
    country          : str
    gender           : str
    age              : float
    tenure           : float
    balance          : float
    products_number  : int
    credit_card      : int
    active_member    : int
    estimated_salary : float

    @validator("credit_score")
    def validate_credit_score(cls, v):
        if not (300 <= v <= 850):
            raise ValueError("Credit score phải từ 300 đến 850")
        return v

    @validator("age")
    def validate_age(cls, v):
        if not (18 <= v <= 100):
            raise ValueError("Age phải từ 18 đến 100")
        return v

    @validator("tenure")
    def validate_tenure(cls, v):
        if not (0 <= v <= 10):
            raise ValueError("Tenure phải từ 0 đến 10")
        return v

    @validator("balance")
    def validate_balance(cls, v):
        if v < 0:
            raise ValueError("Balance không được âm")
        return v

    @validator("products_number")
    def validate_products(cls, v):
        if not (1 <= v <= 4):
            raise ValueError("Products number phải từ 1 đến 4")
        return v

    @validator("credit_card", "active_member")
    def validate_binary(cls, v):
        if v not in (0, 1):
            raise ValueError("Giá trị phải là 0 hoặc 1")
        return v

    @validator("country")
    def validate_country(cls, v):
        if v not in ["France", "Spain", "Germany"]:
            raise ValueError("Country phải là France, Spain hoặc Germany")
        return v

    @validator("gender")
    def validate_gender(cls, v):
        if v not in ["Male", "Female"]:
            raise ValueError("Gender phải là Male hoặc Female")
        return v

    @validator("estimated_salary")
    def validate_salary(cls, v):
        if v <= 0:
            raise ValueError("Estimated salary phải lớn hơn 0")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "credit_score"    : 619,
                "country"         : "France",
                "gender"          : "Female",
                "age"             : 42,
                "tenure"          : 2,
                "balance"         : 0,
                "products_number" : 1,
                "credit_card"     : 1,
                "active_member"   : 1,
                "estimated_salary": 101348.88
            }
        }