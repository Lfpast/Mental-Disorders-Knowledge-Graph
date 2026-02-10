import re

def normalize_span(text: str) -> str:
    value = " ".join(text.strip().split())
    value = value.lower().replace("\t", " ")
    value = re.sub(r"\s+", "_", value)
    return value
