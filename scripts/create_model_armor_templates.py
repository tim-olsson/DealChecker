"""Creates (or updates) the Model Armor templates used by DealChecker's guardrails.

Two templates:
  - deal-checker-input:  screens text sent to the model (user prompts and the
    listing text passed between agents) for prompt injection / jailbreaks and
    harmful content (the malicious URI filter is not available in europe-west2).
  - deal-checker-output: screens model responses for harmful content and
    sensitive data (Sensitive Data Protection basic: e.g. emails, phone
    numbers, card numbers).

Usage (requires `gcloud auth application-default login` and the Model Armor API):
    uv run scripts/create_model_armor_templates.py --project <project-id> --location europe-west2

Idempotent: existing templates are updated to match this file.
Prints the env vars to put in `.env`.
"""

import argparse

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import AlreadyExists
from google.cloud import modelarmor_v1 as ma

INPUT_TEMPLATE_ID = "deal-checker-input"
OUTPUT_TEMPLATE_ID = "deal-checker-output"

RAI_FILTERS = [
    ma.RaiFilterSettings.RaiFilter(filter_type=t, confidence_level=ma.DetectionConfidenceLevel.MEDIUM_AND_ABOVE)
    for t in (
        ma.RaiFilterType.HATE_SPEECH,
        ma.RaiFilterType.HARASSMENT,
        ma.RaiFilterType.DANGEROUS,
        ma.RaiFilterType.SEXUALLY_EXPLICIT,
    )
]

INPUT_FILTERS = ma.FilterConfig(
    rai_settings=ma.RaiFilterSettings(rai_filters=RAI_FILTERS),
    pi_and_jailbreak_filter_settings=ma.PiAndJailbreakFilterSettings(
        filter_enforcement=ma.PiAndJailbreakFilterSettings.PiAndJailbreakFilterEnforcement.ENABLED,
        confidence_level=ma.DetectionConfidenceLevel.MEDIUM_AND_ABOVE,
    ),
)

OUTPUT_FILTERS = ma.FilterConfig(
    rai_settings=ma.RaiFilterSettings(rai_filters=RAI_FILTERS),
    sdp_settings=ma.SdpFilterSettings(
        basic_config=ma.SdpBasicConfig(
            filter_enforcement=ma.SdpBasicConfig.SdpBasicConfigEnforcement.ENABLED,
        )
    ),
)


def upsert(client: ma.ModelArmorClient, parent: str, template_id: str, filters: ma.FilterConfig) -> str:
    name = f"{parent}/templates/{template_id}"
    template = ma.Template(name=name, filter_config=filters)
    try:
        client.create_template(parent=parent, template_id=template_id, template=template)
        print(f"Created template: {name}")
    except AlreadyExists:
        client.update_template(template=template)
        print(f"Updated template: {name}")
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", required=True)
    parser.add_argument("--location", required=True, help="e.g. europe-west2")
    args = parser.parse_args()

    client = ma.ModelArmorClient(
        client_options=ClientOptions(api_endpoint=f"modelarmor.{args.location}.rep.googleapis.com")
    )
    parent = f"projects/{args.project}/locations/{args.location}"
    input_name = upsert(client, parent, INPUT_TEMPLATE_ID, INPUT_FILTERS)
    output_name = upsert(client, parent, OUTPUT_TEMPLATE_ID, OUTPUT_FILTERS)

    print("\nAdd to .env:")
    print(f"MODEL_ARMOR_PROMPT_TEMPLATE={input_name}")
    print(f"MODEL_ARMOR_RESPONSE_TEMPLATE={output_name}")


if __name__ == "__main__":
    main()
