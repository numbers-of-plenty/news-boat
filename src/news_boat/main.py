import os
import datetime
import asyncio
import openai
import yaml
from openai import AsyncOpenAI
from dotenv import load_dotenv
import argparse
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional
import json
import pandas as pd
import re
import random
from collections import defaultdict
from typing import List, Dict
import chromadb

load_dotenv()
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
number_of_news = 20  # Number of news items to curate


def parse_args():
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
        "--skip_language",
        action="store_true",
        help="Skip spanification + Spanish sentence + audio generation (default: False)",
    )
    parser.add_argument( #TODO is not used now
        "--runs",
        type=int,
        default=None,
        help="How many times each agent should run (overrides the value in config file)",
    )
    return parser.parse_args()


async def single_agent_news(agent_name: str, topics: List[str], iteration: int = 1, previous_news: str = "") -> str:
    topics_str = "\n".join(f"- {t}" for t in topics)

    iteration_note = ""
    if iteration > 1:
        iteration_note = f"""
        The agent has already produced the following news in a previous iteration:
        {previous_news}
        If the previous agent didn't put <PRIORITY> tags, your job is to assign priorities according to the PRIORITY RULES below and output the corrected version of the previous news with proper <PRIORITY> tags.
        Otherwise, your task is to now to find additional NON-DUPLICATE news or events that were suggested in the previous output
        or explore follow-ups the previous output hinted at. Especially focus on checking proposed websites and websites that might have events posted or listed. 
        Produce ONLY new, non-duplicate items. Strictly only one news should be included per <NEWS_ITEM> ... </NEWS_ITEM> block.
        """

    prompt = f"""
    Search the web. Focus heavily on the info published in the last 72 hours (3 days) but not limit the search to it. Today is
    {datetime.datetime.now().strftime("%Y-%m-%d")}. The ONLY days that satisfy the "last 3 days" request are:
    - { (datetime.datetime.now() - datetime.timedelta(days=0)).strftime("%Y-%m-%d") }
    - { (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d") }
    - { (datetime.datetime.now() - datetime.timedelta(days=2)).strftime("%Y-%m-%d") }
    - { (datetime.datetime.now() - datetime.timedelta(days=3)).strftime("%Y-%m-%d") }

    Fetch information for the following topics:
    {topics_str}

    {iteration_note}

    Return up to 10 items per topic. Each item must use this format with strict <NEWS_ITEM>/<PRIORITY> tags. Strictly only one news should be included per start-end block:
    <NEWS_ITEM>
    ### headline
    - 2-4 sentence summary
    - publication date YYYY-MM-DD or approximate YYYY-MM if exact date unknown
    - single link (Ensure to return the actual URLs, not only reference IDs.)
    <PRIORITY>numeric_value</PRIORITY>
    </NEWS_ITEM>

    PRIORITY RULES:
    4 - Future events one could attend
    3 - Publication explitely from the last 3 days listed above
    2 - Date unknown but possibly within last 3 days
    1 - Older than 3 days but still recent
    1 - News come from the same website/source as a higher priority news item
    0 - Happened months ago or already ended live events
    0 - The news without any link/url/reference
    0 - Duplicate news covering the same event as another higher priority news item

    Do not ask for any confirmation; search outright. If you didn't check something for some reason, specify it AFTER all the news items.
    """

    response = await client.responses.create(
        model="gpt-5-nano",
        tools=[{"type": "web_search"}],
        reasoning={"effort": "low"},
        input=prompt,
        max_output_tokens=20000,
    )

    filename = f"news_{agent_name}_iter{iteration}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(response.output_text)

    return filename


async def all_agents_fetch_news(config):
    agent_files = {}

    # First iteration
    first_iteration_tasks = []
    agent_names = []
    for agent_name, agent_data in config["agents"].items():
        agent_names.append(agent_name)
        topics = agent_data["topics"]
        first_iteration_tasks.append(single_agent_news(agent_name, topics, iteration=1))
    
    first_iteration_files = await asyncio.gather(*first_iteration_tasks)
    
    for agent_name, file1 in zip(agent_names, first_iteration_files):
        agent_files[agent_name] = [file1]

    # Second iteration
    second_iteration_tasks = []
    for agent_name in agent_names:
        with open(agent_files[agent_name][0], "r", encoding="utf-8") as f:
            previous_news = f.read()
        topics = config["agents"][agent_name]["topics"]
        second_iteration_tasks.append(single_agent_news(agent_name, topics, iteration=2, previous_news=previous_news))
    
    second_iteration_files = await asyncio.gather(*second_iteration_tasks)
    
    for agent_name, file2 in zip(agent_names, second_iteration_files):
        agent_files[agent_name].append(file2)

    # Concatenate all agent files into one raw news file
    all_results = []
    for files in agent_files.values():
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                all_results.append(f.read())

    raw_news_path = "news_raw.txt"
    with open(raw_news_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_results))

    return raw_news_path, agent_files


async def split_raw_news(raw_news_path: str, chunks_json_path: str):
    """
    Split RAW news text into structured JSON items, extracting:
    - full_text (everything between <NEWS_ITEM> and </NEWS_ITEM>)
    - priority (integer from <PRIORITY> tag)
    """

    with open(raw_news_path, "r", encoding="utf-8") as f:
        raw_news = f.read()

    # Regex for each news block
    news_pattern = re.compile(r"<NEWS_ITEM>(.*?)</NEWS_ITEM>", re.DOTALL | re.IGNORECASE)
    news_blocks = news_pattern.findall(raw_news)

    items = []
    for block in news_blocks:
        block_clean = block.strip()
        prio_match = re.search(r"<PRIORITY>(\d+)</PRIORITY>", block_clean, re.IGNORECASE)
        priority = int(prio_match.group(1)) if prio_match else 0
        # Remove priority tag from full_text
        full_text_clean = re.sub(r"<PRIORITY>\d+</PRIORITY>", "", block_clean, flags=re.IGNORECASE).strip()

        items.append({
            "full_text": full_text_clean,
            "priority": priority
        })

    with open(chunks_json_path, "w", encoding="utf-8") as f:
        json.dump({"items": items}, f, ensure_ascii=False, indent=2)

    # legacy compatibility
    class DummyResponse:
        output_parsed = {"items": items}

    return DummyResponse()


async def embed_text(sentences):
    
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=sentences,
    )

    return sentences, response.data[0].embedding


async def shuffle_sort_news(news_items: List[Dict], top_n: int = 40) -> List[Dict]:
    """
    Sorts a list of news items by priority descending, shuffles within each priority group,
    and returns the first `top_n` items.
    
    Args:
        news_items: List of dicts with keys 'full_text' and 'priority'.
        top_n: Number of items to return after sorting and shuffling.
    
    Returns:
        Sorted and shuffled list of news items (max length = top_n).
    """
    # Group news by priority
    priority_groups = defaultdict(list)
    for item in news_items['items']:
        priority_groups[int(item["priority"])].append(item)

    # Shuffle each group
    for items in priority_groups.values():
        random.shuffle(items)

    # Sort priorities descending and flatten
    sorted_news = []
    for prio in sorted(priority_groups.keys(), reverse=True):
        sorted_news.extend(priority_groups[prio])

    # Take top_n
    top_news = sorted_news[:top_n]

    # Write json to the file top_raw_news
    with open("top_raw_news.json", "w", encoding="utf-8") as f:
        json.dump({"items": top_news}, f, ensure_ascii=False, indent=2)

    return top_news  


# async def embed_texts(news_texts):
    
#     tasks = []
#     for text in news_texts:
#         tasks.append(embed_sentences(text))

#     results = await asyncio.gather(*tasks)

#     sentences, embeddings = zip(*results)
#     # Create DataFrame: index = sentence, column = embedding vector
#     embeddings_df = pd.DataFrame({
#         "embedding": embeddings
#     }, index=sentences)

#     embeddings_df.to_csv('news_embeddings.tsv', sep = '\t')
#     return embeddings_df
    

# ---------------------------------------------------

async def find_closest_past_news(news_embed, top_n = 3):

    client = chromadb.PersistentClient(path="./chroma_db") 
    collection = client.get_or_create_collection(name="df_news_embeddings")
    results = collection.query(
        query_embeddings=[news_embed],
        n_results=3,
        include=['documents']
    )
  
    return results['documents'][:top_n]


class DuplicateCheckResponse(BaseModel):
    is_duplicate: bool = Field(
        description="True if the news is a duplicate of any past news, False if it's unique/new"
    )


async def llm_check_duplicate(news_text) -> bool:
    """
    Check if a news item is a duplicate of past news.
    Returns True if NOT a duplicate (unique), False if it IS a duplicate.
    """
    _, news_embedding = await embed_text(news_text)
    
    # Get the 3 most similar past news items
    past_news_list = await find_closest_past_news(news_embedding, top_n=3)
    
    # Flatten the list if needed (chromadb returns nested structure)
    if past_news_list and isinstance(past_news_list[0], list):
        past_news_list = past_news_list[0]
    
    # Format past news for the prompt
    past_news_formatted = "\n\n---\n\n".join(
        [f"Past News {i+1}:\n{news}" for i, news in enumerate(past_news_list)]
    ) if past_news_list else "No past news found"
    
    prompt = f"""
    Determine if the CURRENT NEWS is a duplicate of any of the PAST NEWS items.
    
    A news item is considered a DUPLICATE if:
    - It covers the exact same event/announcement
    - It's about the same release/product with the same release date
    - It's substantially the same story from a different source
    
    A news item is NOT a duplicate if:
    - It's about a different event, even in the same category
    - It's a follow-up or update with new information
    - It's about a different episode/season/version of a series/game
    - The dates or details are significantly different
    
    CURRENT NEWS:
    {news_text}
    
    PAST NEWS:
    {past_news_formatted}
    
    Respond with your assessment.
    """
    
    response = await client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a news duplicate detection system."},
            {"role": "user", "content": prompt}
        ],
        response_format=DuplicateCheckResponse,
    )
    
    result = response.choices[0].message.parsed
    
    # Return True if NOT duplicate, False if IS duplicate
    return not result.is_duplicate, news_embedding

async def remove_duplicates(top_news):
    """
    Remove duplicate news items by checking each against past news.
    Returns fresh (non-duplicate) news and their embeddings.
    Writes fresh news to fresh_news.txt in raw format compatible with curate_news.
    """
    tasks = []
    for news_item in top_news:
        news_text = news_item['full_text']
        tasks.append(llm_check_duplicate(news_text))
    
    results = await asyncio.gather(*tasks)
    
    fresh_news = []
    embeddings = []
    
    i = 0 #add 11 news items maximum
    for news_item, (is_not_duplicate, news_embedding) in zip(top_news, results):
        if is_not_duplicate: #filter duplicates
            i += 1
            fresh_news.append(news_item)
            embeddings.append(news_embedding)
            if i > 10:
                break
    
    fresh_news_raw = "\n\n".join([
        f"{item['full_text']}\n\n" for item in fresh_news
    ])
    
    with open("fresh_news.txt", "w", encoding="utf-8") as f:
        f.write(fresh_news_raw)
    
    return fresh_news, embeddings


    

async def write_text_vectors(fresh_news, embeddings):
    """
    Write fresh news items and their embeddings to ChromaDB.
    """
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="df_news_embeddings")
    
    # Prepare data for ChromaDB
    ids = [f"news_{i}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}" for i in range(len(fresh_news))]
    documents = [item['full_text'] for item in fresh_news]
    
    # Add to ChromaDB
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents
    )
    
    print(f"Added {len(fresh_news)} news items to ChromaDB")



async def curate_news(raw_news_path: str, output_path: str) -> str:
    """
    Process raw news data to create a curated list of top 11 impactful news items.
    """
    with open(raw_news_path, "r", encoding="utf-8") as f:
        raw_news = f.read()

    # Build agent topics context

    prompt = f"""
    You are given raw news/events data collected by multiple agents. Process this data according to these requirements:
    
    1. Delete any duplicate news items covering the same event. Leave only one such news item.
    2. Check that the news item has the actual date. Try to find the real date if it's wrong.
    3. Check that the link/URL is valid and points to the actual news source. If not, try to find a valid link with the news. If you can't find a valid link, discard the news item.
    
    OUTPUT FORMAT:
    4. Format each news item with this exact structure so that they were uniform:
       
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
        reasoning={"effort": "low"},
        tools=[{"type": "web_search"}],      
        input=prompt,
        max_output_tokens=25000,
    )

    curated_news = response.output_text

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(curated_news)

    return curated_news


def print_section(message):
    """Print a message wrapped with separator lines."""
    print("\n" + "=" * 50)
    print(message)
    print("=" * 50 + "\n")


async def main():

    print_section("Fetching news according to topics in the config")
    raw_news_path, agent_files = await all_agents_fetch_news(config)

    print_section("Splitting RAW news into structured chunks...")
    raw_chunks = await split_raw_news("news_raw.txt", "news_raw_separate_items.json")
    list_of_news = raw_chunks.output_parsed

    print_section("Selecting 40 news with highest priority")
    top_news = await shuffle_sort_news(list_of_news, 40)

    print_section('projecting news items into embeddings...')
    fresh_news, embeddings = await remove_duplicates(top_news)

    print_section('Writing embeddings to ChromaDB...')
    await write_text_vectors(fresh_news, embeddings)

    print_section("Curating news...")
    curated_news = await curate_news("fresh_news.txt", "news_curated.txt")

    print("\nfinished!")


if __name__ == "__main__":
    args = parse_args()

    config_path = os.path.abspath(args.config)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    asyncio.run(main())
