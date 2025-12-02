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
number_of_news = 11


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


async def single_agent_news(agent_name: str, iteration: int = 1, previous_news: str = "") -> str:
    topics = config["agents"][agent_name]["topics"]
    topics_str = "\n".join(f"- {t}" for t in topics)

    iteration_note = ""
    if iteration > 1:
        iteration_note = f"""
        The agent has already produced the following news in a previous iteration:
        {previous_news}
        If the previous agent didn't put <PRIORITY> tags, your job is to assign priorities according to the PRIORITY RULES below.
        Otherwise, your task is to now to find additional news or events that were suggested in the previous output
        or explore follow-ups the previous output hinted at. Especially focus on checking proposed websites and websites that might have events posted or listed. 
        Produce ONLY new, non-duplicate items. Strictly only one news should be included per start-end block.
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

    Return up to 10 items per topic. Each item must use this format with strict <START>/<END>/<PRIORITY> tags. Strictly only one news should be included per start-end block:
    <START>
    ### headline
    - 2-4 sentence summary
    - publication date YYYY-MM-DD or approximate YYYY-MM if exact date unknown
    - single link (Ensure to return the actual URLs, not only reference IDs.)
    <PRIORITY>numeric_value</PRIORITY>
    <END>

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


# async def single_agent_news(agent_name: str) -> str:
#     topics = config["agents"][agent_name]["topics"]
#     topics_str = "\n".join(f"- {t}" for t in topics)

#     prompt = f"""
#     Search the web. Focus heavily on the info published in the last 72 hours (3 days) but not limit the search to it. Today is
#     {datetime.datetime.now().strftime("%Y-%m-%d")}. The days that satisfy the "last 3 days" request are:
#     - { (datetime.datetime.now() - datetime.timedelta(days=0)).strftime("%Y-%m-%d") }
#     - { (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d") }
#     - { (datetime.datetime.now() - datetime.timedelta(days=2)).strftime("%Y-%m-%d") }
#     - { (datetime.datetime.now() - datetime.timedelta(days=3)).strftime("%Y-%m-%d") }
    
#     Fetch information for the following topics:
#     {topics_str}

#     Return up to 10 items per topic.
#     Each item should be formatted as below with <START>,<END> and <PRIORITY>,</PRIORITY> strictly included for subsequent parsing. Strictly only one news should be included per start-end block.:
#     <START>
#     ### headline
#     - 2-4 sentence summary
#     - publication date YYYY-MM-DD format or approximate YYYY-MM if exact date unknown
#     - OR the date of the most recent event described on the website if the publication date is not available (must be inside the last 72 hours or in the future).
#     - single link (Ensure to return the actual URLs, not only reference IDs.)
#     <PRIORITY>numeric_value</PRIORITY>
#     <END>

#         PRIORITY RULES (must use only integer values):
#     4 - Future events one could attend
#     3 - Publication explitely from the last 3 days
#     2 - Date unknown but possibly within last 3 days
#     1 - Older than 3 days but still recent
#     1 - News come from the same website/source as a higher priority news item
#     0 - Happened months ago or already ended live events
#     0 - The news without any link/url/reference
#     0 - Duplicate news covering the same event as another higher priority news item

#     Do not ask me for any confirmations or permissions, search outright.
#     """

#     response = await client.responses.create(
#         model="gpt-5-nano",
#         tools=[{"type": "web_search"}],
#         reasoning={"effort": "small"},
#         input=prompt,
#         max_output_tokens=20000,
#     )
#     return f"--- {agent_name} ---\n" + response.output_text


async def all_agents_fetch_news():
    agent_files = {}

    # First iteration
    for agent in config["agents"].keys():
        file1 = await single_agent_news(agent, iteration=1)
        agent_files.setdefault(agent, []).append(file1)

    # Second iteration
    for agent in config["agents"].keys():
        with open(agent_files[agent][0], "r", encoding="utf-8") as f:
            previous_news = f.read()
        file2 = await single_agent_news(agent, iteration=2, previous_news=previous_news)
        agent_files[agent].append(file2)

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
    - full_text (everything between <START> and <END>)
    - priority (integer from <PRIORITY> tag)
    """

    with open(raw_news_path, "r", encoding="utf-8") as f:
        raw_news = f.read()

    # Regex for each news block
    news_pattern = re.compile(r"<START>(.*?)<END>", re.DOTALL | re.IGNORECASE)
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
    agent_topics_context = []
    for agent_name, agent_data in config["agents"].items():
        topics_list = ", ".join(agent_data["topics"])
        agent_topics_context.append(f"- {agent_name}: {topics_list}")
    agent_topics_str = "\n".join(agent_topics_context)

    prompt = f"""
    You are given raw news/events data collected by multiple agents. Process this data according to these requirements:
    
    FILTERING RULES:

    1. All events which can be visited in the future get a free pass (should be 100% kept). Others are subject to filtering.
    2. Discard any news items that do NOT have any kind of a link/url (or at least refernce)

    PRIORITY RULES:
    3. Give higher priority to the news explicitly marked as being from the last 72 hours (last 3 days)
    4. Give higher priority to news that are the most impactful in their respective topic.
    5. Give medium priority to the news without explicit date but which are likely recent.
    6. Give the low priority to the news which explicitly are older than 7 days. 
    7. Give the low priority to the local events which already ended in the previous days.
    8. If several news items have the same link / source website, choose just one of them to give it a higher priority and give low priority to all the other news items with the same link/source.
    9. If several news items cover the same event/news, give just one of them a higher priority and give the lowest priority to all the other news item covering the same event.
    10. All events which can be visited in person in the future get the high priority. 

    
    SELECTION RULES:
    9. Select exactly the {number_of_news} most prioritized news / event announcements. Select lower priority news if it's necessary to reach {number_of_news}

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
    1. In EACH news item, replace exactly one word in the summary with its Spanish translation but NOT in the headline
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


async def make_spanish_sentence(spanified_path: str, sentence_txt: str) -> str:
    """
    Extract Spanish words and IPA from spanified news,
    generate the shortest meaningful Spanish text using all of them,
    and save the sentence to a .txt file.
    """
    with open(spanified_path, "r", encoding="utf-8") as f:
        spanified = f.read()

    prompt = f"""
    You are given text where Spanish words appear in each news item. 

    TASKS:
    1. Extract all UNIQUE Spanish words
    2. Using ALL of these words, create the SHORT Spanish text
       that is still grammatical and makes sense. Do not reuse news content.
    3. Return ONLY the text, without any additional commentary.


    INPUT:
    {spanified}
    """

    response = await client.responses.create(
        model="gpt-5-nano",
        reasoning={"effort": "low"},
        input=prompt,
        max_output_tokens=4000,
    )

    result = response.output_text.strip()
    sentence = result.split("\n")[0].strip()

    # save sentence
    with open(sentence_txt, "w", encoding="utf-8") as f:
        f.write(sentence)

    return sentence


async def generate_audio(sentence: str, audio_path: str):
    """
    Generate speech audio for the given Spanish sentence.
    Saves the output to an MP3 file.
    """

    with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=sentence,
    ) as response:
        response.stream_to_file(audio_path)

    return response


def print_section(message):
    """Print a message wrapped with separator lines."""
    print("\n" + "=" * 50)
    print(message)
    print("=" * 50 + "\n")


async def main():

    print_section("Fetching news according to topics in the config")
    raw_news_path, agent_files = await all_agents_fetch_news()

    print_section("Splitting RAW news into structured chunks...")
    raw_chunks = await split_raw_news("news_raw.txt", "news_raw_separate_items.json")
    list_of_news = raw_chunks.output_parsed

    print_section("Selecting 40 news with highest priority")
    top_news = await shuffle_sort_news(list_of_news, 10)

    print_section('projecting news items into embeddings...')
    fresh_news, embeddings = await remove_duplicates(top_news)

    print_section('Writing embeddings to ChromaDB...')
    await write_text_vectors(fresh_news, embeddings)

    print_section("Curating news...")
    curated_news = await curate_news("fresh_news.txt", "news_curated.txt")

    # flag to avoid language learning steps
    if args.skip_language:
        print("Language processing disabled (--language False).")
        print("Generated: news_raw.txt and news_curated.txt")
        return

    print_section("Spanifying news...")
    spanified_news = await spanify_news("news_curated.txt", "news_spanified.txt")
    # print(spanified_news)

    print_section("Creating Spanish vocabulary sentence...")
    sentence = await make_spanish_sentence("news_spanified.txt", "spanish_sentence.txt")

    print_section("Generating audio...")
    await generate_audio(sentence, "spanish_sentence.mp3")

    print("Sentence saved to spanish_sentence.txt")
    print("Audio saved to spanish_sentence.mp3")

    print("\nfinished!")


if __name__ == "__main__":
    args = parse_args()

    config_path = os.path.abspath(args.config)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cli_runs = args.runs
    config_runs = config.get("runs", 2)
    runs = cli_runs if cli_runs is not None else config_runs

    asyncio.run(main())
