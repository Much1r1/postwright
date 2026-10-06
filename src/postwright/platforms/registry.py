from postwright.platforms.base import PlatformAdapter
from postwright.platforms.linkedin_adapter import LinkedInAdapter
from postwright.platforms.x_adapter import XAdapter


def get_adapter(platform: str) -> PlatformAdapter:
    """Return platform adapter instance for given platform name."""
    norm = platform.lower().strip()
    if norm in ("x", "twitter"):
        return XAdapter()
    elif norm == "linkedin":
        return LinkedInAdapter()
    else:
        raise ValueError(f"Unsupported platform: '{platform}'")
