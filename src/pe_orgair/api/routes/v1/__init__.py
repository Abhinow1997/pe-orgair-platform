from fastapi import APIRouter

router = APIRouter(tags=["v1"])

@router.get("/")
async def v1_root():
    return {"message": "V1 API", "version": "1.0"}