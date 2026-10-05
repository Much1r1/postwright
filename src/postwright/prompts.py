from pathlib import Path

VOICE_DIR = Path(__file__).parent.parent.parent / "voice"


def get_voice_profile() -> str:
    profile_path = VOICE_DIR / "profile.md"
    if profile_path.exists():
        return profile_path.read_text(encoding="utf-8")
    return ""


def get_voice_examples() -> str:
    examples_dir = VOICE_DIR / "examples"
    if not examples_dir.exists():
        return ""
    examples = []
    for file_path in sorted(examples_dir.glob("*.md")):
        examples.append(file_path.read_text(encoding="utf-8"))
    return "\n\n---\n\n".join(examples)
