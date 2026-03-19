import httpx
from PIL import Image
from PIL import Bytesio

MAX_BYTES = 5_000_000 #5 MB maximum
TIMEOOUT_S = 6.0

async def fetch_image(url: str) -> Image.Image:
    async with httpx.AsyncClient(timeout=TIMEOOUT_S, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
        content = r.content
        
        if len(content) > MAX_BYTES:
            raise ValueError(f"Image provided is too large")
    
    img = Image.open(Bytesio(content)).convert("RGB")
    return img