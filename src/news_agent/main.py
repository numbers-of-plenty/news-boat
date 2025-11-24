import os
import datetime
import asyncio
import yaml
from openai import AsyncOpenAI
from dotenv import load_dotenv
import argparse

load_dotenv()
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


parser = argparse.ArgumentParser(
    description="Run news agent with configurable config file"
)
parser.add_argument(
    "--config",
    type=str,
    default="config.yaml",
    help="Path to config YAML file (default: config.yaml)",
)
parser.add_argument(
    "--runs",
    type=int,
    default=None,
    help="How many times each agent should run (overrides the value in config file)",
)
args = parser.parse_args()
number_of_news = 11
config_path = os.path.abspath(args.config)

with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

cli_runs = args.runs
config_runs = config.get("runs", 2)
runs = cli_runs if cli_runs is not None else config_runs


async def fetch_agent_news(agent_name: str) -> str:
    topics = config["agents"][agent_name]["topics"]
    topics_str = "\n".join(f"- {t}" for t in topics)

    prompt = f"""
    Search the web. Focus heavily on the info published in the last 72 hours (3 days) but not limit the search to it. Today is
    {datetime.datetime.now().strftime("%Y-%m-%d")}.
    
    Fetch information for the following topics:
    {topics_str}

    Return up to 10 items per topic.
    Each item should have:
    - headline
    - 2-4 sentence summary
    - publication date (must be inside the last 72 hours) 
    - OR the date of the most recent event described on the website if the publication date is not available (must be inside the last 72 hours or in the future).
    - link (Return actual URLs, not only reference IDs.)

    Do not ask me for any confirmations or permissions, search outright.
    """

    response = await client.responses.create(
        model="gpt-5-nano",
        tools=[{"type": "web_search"}],
        reasoning={"effort": "low"},
        input=prompt,
        max_output_tokens=10000,
    )
    return f"--- {agent_name} ---\n" + response.output_text


async def curate_news(raw_news_path: str, output_path: str) -> str:
    """
    Process raw news data to create a curated list of top 11 impactful news items.
    """
    with open(raw_news_path, "r", encoding="utf-8") as f:
        raw_news = f.read()

    # Build agent topics context
    agent_topics_context = []
    for agent_name, agent_data in config["agents"].items():
        topics_list = ", ".join(agent_data["topics"])
        agent_topics_context.append(f"- {agent_name}: {topics_list}")
    agent_topics_str = "\n".join(agent_topics_context)

    prompt = f"""
    You are given raw news/events data collected by multiple agents. Process this data according to these requirements:
    
    FILTERING RULES:

    1. All events which can be visited in the future get a free pass. Others are subject to filtering.
    2. Discard any news items that do NOT have any kind of a link/url (or at least refernce)

    PRIORITY RULES:
    3. Give higher priority to the news explicitly marked as being from the last 72 hours (last 3 days)
    4. Give higher priority to news that are the most impactful in their respective topic.
    5. Give medium priority to the news without explicit date but which are likely recent.
    6. Give the low priority to the news which explicitly are older than 7 days. 
    7. Give the low priority to the local events which already ended in the previous days.
    8. Give the low priority to the news whose link is duplicate / leads to the same website as other news. 

    
    SELECTION RULES:
    8. Select exactly the {number_of_news} most prioritized news / event announcements. Select lower priority news if it's necessary to reach {number_of_news}

    Agent topic areas:
    {agent_topics_str}
    
    Today's date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    
    OUTPUT FORMAT:
    9. Format each news item with this exact structure:
       
       ## [Headline]
       [Several sentence summary copy-pasted from the given news]    
       Date:[publication date or most recent event date. Date must be only here]
       Link:[URL]
       
       ---
    
    RAW NEWS DATA:
    {raw_news}
    
    Provide only the curated news items in the format specified above. Do not include any additional commentary.
    """

    response = await client.responses.create(
        model="gpt-5-nano",
        reasoning={"effort": "medium"},
        input=prompt,
        max_output_tokens=25000,
    )

    curated_news = response.output_text

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(curated_news)

    return curated_news


async def spanify_news(curated_news_path: str, output_path: str) -> str:
    """
    Process curated news to replace one word per sentence with Spanish,
    add IPA transliteration, and include explanations.
    """
    with open(curated_news_path, "r", encoding="utf-8") as f:
        curated_news = f.read()

    prompt = f"""
    You are given curated news items. Transform them by replacing EXACTLY ONE word per news item with its Spanish equivalent.
    
    REPLACEMENT RULES:
    1. In EACH sentence, replace exactly one word with its Spanish translation
    2. Use the most basic form: nominative/singular for nouns, infinitive for verbs
    3. Immediately after the Spanish word, add its IPA transliteration in square brackets
    4. Vary the types of words replaced (nouns, verbs, adjectives, etc.)
    
    After each news add the vocabulary for the words that were just replaced. It should include the word, its IPA transliteration, English sentence defining the word and the English translation itself. E.G.
    **Vocabulary:**
    - **[Gatto]** /[ˈɡat.to]/ - [A fluffy, relatively small predator which is popular as a pet. Cat.]
    ---
    
    CURATED NEWS:
    {curated_news}
    
    Provide the transformed news items with Spanish word replacements, IPA transliterations in square brackets immediately after each Spanish word, and vocabulary explanations below each item. Do NOT include any additional commentary.
    """

    response = await client.responses.create(
        model="gpt-5-nano",
        reasoning={"effort": "medium"},
        input=prompt,
        max_output_tokens=40000,
    )

    spanified_news = response.output_text

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(spanified_news)

    return spanified_news


async def main():
    tasks = []
    for _ in range(runs):  # 2 synchronous runs per agent by default
        for agent in config["agents"].keys():
            tasks.append(fetch_agent_news(agent))

    print("Fetching news according to topics in config.yaml")
    all_results = await asyncio.gather(*tasks)
    raw_news = "\n\n".join(all_results)
    # print(final_output)

    raw_news_path = "news_raw.txt"
    with open(raw_news_path, "w", encoding="utf-8") as f:
        f.write(raw_news)

    print("\n" + "=" * 50)
    print("Curating news...")
    print("=" * 50 + "\n")

    curated_news = await curate_news(raw_news_path, "news_curated.txt")
    # print(curated_news)

    print("\n" + "=" * 50)
    print("Spanifying news...")
    print("=" * 50 + "\n")

    spanified_news = await spanify_news("news_curated.txt", "news_spanified.txt")
    # print(spanified_news)

    print('\nfinished!')


if __name__ == "__main__":
    asyncio.run(main())
