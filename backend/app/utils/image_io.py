from io import BytesIO
import asyncio
import ipaddress
import socket
from urllib.parse import urlparse

import httpx
from PIL import Image

from app.config import get_settings

settings = get_settings()


async def _validate_remote_image_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Image URL must use http or https and include a host")

    # Rejecting private-network targets keeps the API from becoming a proxy
    # into localhost or internal services when the extension sends user input.
    if settings.allow_private_image_hosts:
        return

    host = parsed.hostname
    if host is None:
        raise ValueError("Image URL is missing a valid hostname")

    addrinfo = await asyncio.to_thread(socket.getaddrinfo, host, None)
    for _, _, _, _, sockaddr in addrinfo:
        ip = ipaddress.ip_address(sockaddr[0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValueError("Image URL resolves to a private or unsupported host")


async def fetch_image(url: str) -> Image.Image:
    await _validate_remote_image_url(url)

    async with httpx.AsyncClient(
        timeout=settings.fetch_timeout_s,
        follow_redirects=True,
        trust_env=False,
    ) as client:
        response = await client.get(url)
        response.raise_for_status()

        # Size and content-type checks protect the backend, and keep response
        # times predictable for the extension client.
        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            raise ValueError("URL did not return an image resource")

        content = response.content
        if len(content) > settings.max_image_bytes:
            raise ValueError("Image provided is too large")

    return Image.open(BytesIO(content)).convert("RGB")
