import pydantic as BaseModel

class ConfigureAvatar(BaseModel.BaseModel):
    hairId: str
    hairColorId: str
    skinColorId: str
    facialHairId: str
    clothingId: str
