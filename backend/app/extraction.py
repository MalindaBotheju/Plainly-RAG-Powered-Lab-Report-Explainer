"""
Extraction step.

Takes the raw bytes of an uploaded PNG/JPG lab report image and asks a
Groq vision model to return a strict JSON list of test rows. We
deliberately do NOT ask the model to interpret anything here -- this step
only reads what's on the page, so errors are easy to isolate to either
"extraction" or "interpretation".

Uses Groq's vision model (see config.GROQ_VISION_MODEL), not Gemini --
switched to cut daily free-tier request costs and drop a deprecated SDK
dependency (google-generativeai). PDF support was dropped in this switch
since Groq's vision model only accepts images; see config.py's comment on
ALLOWED_MIME_TYPES for why.
"""
import base64
import json
import re
from typing import List

from groq import Groq

from . import config
from .schemas import LabValue

_EXTRACTION_PROMPT = """You are reading a medical lab report (image or PDF page).

Extract every test result row you can find into a JSON array. For each row output:
{
  "test": "<test name exactly as printed>",
  "value": "<numeric result as printed>",
  "unit": "<unit as printed, empty string if none>",
  "reference_range": "<reference/normal range as printed, empty string if none>",
  "reported_flag": "<any flag the report itself prints, e.g. H, L, HIGH, LOW, empty string if none>"
}

Rules:
- Only extract rows that are actual test results with a numeric or clearly measurable value.
- Do not compute or infer anything (no interpretation, no flags you invent yourself).
- Do not include patient name, address, or any other identifying information in the output.
- Return ONLY the JSON array, no markdown fences, no commentary, no extra text.
- If you cannot find any test result rows, return an empty JSON array: []
"""


def _get_client() -> Groq:
    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at "
            "https://console.groq.com/keys and set it in your .env "
            "(local) or Render environment variables (deployed)."
        )
    return Groq(api_key=config.GROQ_API_KEY)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def extract_lab_values(file_bytes: bytes, mime_type: str) -> List[LabValue]:
    """
    Send the image straight to Groq's vision model as a base64 data URI
    and parse the JSON array it returns into LabValue objects.
    """
    client = _get_client()

    b64_data = base64.b64encode(file_bytes).decode("utf-8")
    data_uri = f"data:{mime_type};base64,{b64_data}"

    completion = client.chat.completions.create(
        model=config.GROQ_VISION_MODEL,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _EXTRACTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
    )

    raw = _strip_code_fences(completion.choices[0].message.content or "[]")

    try:
        rows = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model did not return valid JSON for extraction. "
            f"Raw output: {raw[:500]}"
        ) from e

    values: List[LabValue] = []
    for row in rows:
        values.append(
            LabValue(
                test=str(row.get("test", "")).strip(),
                value=str(row.get("value", "")).strip(),
                unit=str(row.get("unit", "")).strip(),
                reference_range=str(row.get("reference_range", "")).strip(),
                reported_flag=str(row.get("reported_flag", "")).strip(),
            )
        )
    return [v for v in values if v.test and v.value]
