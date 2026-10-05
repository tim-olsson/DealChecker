"""Creates (or finds) the Vertex AI Agent Engine instance used by DealChecker.

The instance provides managed Sessions and Memory Bank. No agent code is
deployed to it; the agent still runs locally (`src/main.py` / `adk web`).

Usage (requires `gcloud auth application-default login`):
    uv run scripts/create_agent_engine.py --project <project-id> --location europe-west2

Idempotent: if an instance with the same display name exists, it is reused.
Prints the resource name to put in `.env` as SESSION_SERVICE_URI / MEMORY_SERVICE_URI.
"""

import argparse

import vertexai

DISPLAY_NAME = "deal-checker"

MEMORY_TOPICS = [
    {"managed_memory_topic": {"managed_topic_enum": "USER_PREFERENCES"}},
    {"managed_memory_topic": {"managed_topic_enum": "EXPLICIT_INSTRUCTIONS"}},
    {
        "custom_memory_topic": {
            "label": "shopping_goals",
            "description": (
                "Products the user is looking for, their budget or price limits, "
                "preferred condition, must-have features and deal-breakers."
            ),
        }
    },
    {
        "custom_memory_topic": {
            "label": "deal_outcomes",
            "description": (
                "Listings the user checked: product, asking price and currency, "
                "condition, verdict, and what the user decided (bought, offered, "
                "skipped). Do not store seller names or contact details."
            ),
        }
    },
]


def find_existing(client: vertexai.Client) -> str | None:
    for engine in client.agent_engines.list():
        if engine.api_resource.display_name == DISPLAY_NAME:
            return engine.api_resource.name
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", required=True)
    parser.add_argument("--location", required=True, help="e.g. europe-west2")
    args = parser.parse_args()

    client = vertexai.Client(project=args.project, location=args.location)

    name = find_existing(client)
    if name:
        print(f"Reusing existing Agent Engine: {name}")
    else:
        engine = client.agent_engines.create(
            config={
                "display_name": DISPLAY_NAME,
                "description": "DealChecker sessions and Memory Bank.",
                "context_spec": {
                    "memory_bank_config": {
                        "customization_configs": [{"memory_topics": MEMORY_TOPICS}],
                    }
                },
            }
        )
        name = engine.api_resource.name
        print(f"Created Agent Engine: {name}")

    print("\nAdd to .env:")
    print(f"SESSION_SERVICE_URI=agentengine://{name}")
    print(f"MEMORY_SERVICE_URI=agentengine://{name}")


if __name__ == "__main__":
    main()
