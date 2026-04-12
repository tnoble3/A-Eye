from io import BytesIO
import httpx
from PIL import Image

MAX_BYTES = 5_000_000
TIMEOUT_S = 6.0


async def fetch_image(url: str) -> Image.Image:
    async with httpx.AsyncClient(timeout=TIMEOUT_S, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
#size and content type checks protect the backend, and keep response times predictable for the extension client
        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            raise ValueError("URL did not return an image resource")
        content = response.content
        if len(content) > MAX_BYTES:
            raise ValueError("Image provided is too large")
    return Image.open(BytesIO(content)).convert("RGB")
