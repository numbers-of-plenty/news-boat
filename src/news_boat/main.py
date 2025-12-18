import os
import datetime
import asyncio
import yaml
from openai import AsyncOpenAI
from dotenv import load_dotenv
import argparse
from pydantic import BaseModel, Field
from typing import List
import json
import re
import random
from collections import defaultdict
from typing import Dict
import chromadb
from telethon import TelegramClient

load_dotenv()
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
    parser.add_argument(  # TODO is not used now
        "--runs",
        type=int,
        default=None,
        help="How many times each agent should run (overrides the value in config file)",
    )
    parser.add_argument(
        "--use_telegram_api",
        action="store_true",
        help="Use the Telegram API for sending news updates (default: False)",
    )
    parser.add_argument(
        "--api_hash",
        type=str,
        default=None,
        help="Telegram API hash (default: reads from .env file)",
    )
    parser.add_argument(
        "--api_id",
        type=int,
        default=None,
        help="Telegram API ID (default: reads from .env file)",
    )

    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="OpenAI API key (reads from .env file if not provided). If argument's value is False, then OpenAI API won't be used.",
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


async def get_telegram_context(
    config, write_to_path: str = None, api_hash: str = None, api_id: int = None
):
    load_dotenv()
    # Use provided arguments or fall back to environment variables
    if api_hash is None:
        api_hash = os.environ["API_HASH"]
    if api_id is None:
        api_id = int(os.environ["API_ID"])

    # Calculate cutoff time (72 hours ago)
    cutoff_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        hours=72
    )

    all_messages = []
    group_ids = config.get("telegram_group_ids", [])

    async with TelegramClient("session", api_id, api_hash) as client:
        for group_id in group_ids:
            try:
                print(f"Fetching messages from group {group_id}...")
                message_count = 0
                async for msg in client.iter_messages(group_id, limit=200):
                    # Filter for messages from last 72 hours
                    if msg.date and msg.date >= cutoff_time and msg.message:
                        all_messages.append(
                            {
                                "group_id": group_id,
                                "date": msg.date,
                                "message": msg.message,
                            }
                        )
                        message_count += 1
                print(f"  Found {message_count} messages from last 72 hours")
            except Exception as e:
                print(f"Error fetching from group {group_id}: {e}")

    # Sort by date
    all_messages.sort(key=lambda x: x["date"], reverse=True)

    joined_messages = "\n\n".join(
        [
            f"Date: {msg_data['date'].strftime('%Y-%m-%d %H:%M:%S')}\nMessage:\n{msg_data['message']}"
            for msg_data in all_messages
        ]
    )

    # Write to file
    if write_to_path:
        with open(write_to_path, "w", encoding="utf-8") as f:
            f.write(joined_messages)

    # print(f"Wrote {len(all_messages)} messages to {output_file}")
    return all_messages


async def extract_telegram_news(config, telegram_context) -> str:
    """
    Extract news items from Telegram messages using LLM to analyze the context.

    Args:
        config: Config dict containing agent topics
        telegram_context_file: Path to the file containing Telegram messages

    Returns:
        Filename of the output file containing extracted news in <NEWS_ITEM> format
    """

    # Build topics string from all agents
    all_topics = []
    for agent_name, agent_data in config["agents"].items():
        all_topics.extend(agent_data["topics"])
    topics_str = "\n".join(f"- {t}" for t in all_topics)

    prompt = f"""
    You are given messages from Telegram groups from the last 72 hours. Analyze these messages and extract any relevant news, events, or announcements that match the following topics:
    
    {topics_str}
    
    # Today is {datetime.datetime.now().strftime("%Y-%m-%d")}.
    
    For each relevant news item found in the Telegram messages, format it using this strict structure:
    <NEWS_ITEM>
    ### headline, in one sentence, straight to the point, no ":" or "-" characters, just the news itself
    - 1-2 sentence summary based on the Telegram message content. No "-" it should be a single paragraph. Must be concise and give only additional info beyond the headline.
    - message date YYYY-MM-DD (extract from the message date if available)
    - if a link/URL is mentioned in the message, include it; otherwise write "Source: [the name of the group starting with @ in the end of the message]]"
    <PRIORITY>numeric_value</PRIORITY>
    </NEWS_ITEM>

    Telegram source of the message can be found as the next word starting with "@" after the message content. Might be a little bit further in the text but absolutely always present.
    
    FILTERING RULES:
    - Do NOT include somebody's personal opinions on some event (no meta), only hard news
    - Do NOT include that something continues or is ongoing. It only matters if something started or will happen in the future

    PRIORITY RULES:
    4 - Future events one could attend
    3 - News/announcements from the last 3 days
    2 - Date unclear but possibly recent
    1 - Older discussions or past events
    0 - Off-topic or irrelevant to the listed topics
    
    TELEGRAM CONTEXT:
    {telegram_context}
    
    Extract all relevant news items following the format above, in English.
    """

    response = await client.chat.completions.create(
        model="gpt-5-mini",
        reasoning_effort="medium",
        messages=[
            {
                "role": "system",
                "content": "You are a news extraction assistant that analyzes Telegram messages and extracts relevant news items.",
            },
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=10000,
    )

    output_text = response.choices[0].message.content

    filename = "news_raw_telegram.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"Extracted Telegram news to {filename}")
    return output_text


def get_future_dates() -> str:
    """
    Generate a formatted string of future dates for the next 7 days, 3 months, and next year.

    Returns:
        Formatted string with future dates
    """
    today = datetime.datetime.now()

    # Next 7 days
    next_7_days = [
        (today + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 8)
    ]

    # Next 3 months (without specific days)
    next_3_months = []
    for i in range(1, 4):
        future_month = today.month + i
        future_year = today.year

        # Handle year rollover
        while future_month > 12:
            future_month -= 12
            future_year += 1

        next_3_months.append(f"{future_year}-{future_month:02d}")

    # Next year
    next_year = str(today.year + 1)

    # Format output
    future_dates = "- " + "\n- ".join(next_7_days)
    future_dates += "\n- " + "\n- ".join(next_3_months)
    future_dates += f"\n- {next_year}"

    return future_dates


async def single_agent_news(
    agent_name: str, topics: List[str], iteration: int = 1, previous_news: str = ""
) -> str:
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

    future_dates_str = get_future_dates()

    prompt = f"""
    Search the web. Focus heavily on the info published in the last 72 hours (3 days) but not limit the search to it. Today is
    {datetime.datetime.now().strftime("%Y-%m-%d")}. The ONLY days that satisfy the "last 3 days" request are:
    - {(datetime.datetime.now() - datetime.timedelta(days=0)).strftime("%Y-%m-%d")}
    - {(datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")}
    - {(datetime.datetime.now() - datetime.timedelta(days=2)).strftime("%Y-%m-%d")}
    - {(datetime.datetime.now() - datetime.timedelta(days=3)).strftime("%Y-%m-%d")}

    As for future events, here are the dates that clearly lie in the future:
    {future_dates_str}


    Fetch information for the following topics:
    {topics_str}

    {iteration_note}

    Return up to 10 items per topic. Each item must use this format with strict <NEWS_ITEM>/<PRIORITY> tags. Strictly only one news should be included per start-end block:
    <NEWS_ITEM>
    ### headline, in one sentence, straight to the point, no ":" or "-" characters, don't mention the category/subcategory, just the news itself. 
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
    # aid band for hitting rate limits
    filename = f"news_{agent_name}_iter{iteration}.txt"

    try:
        response = await client.responses.create(
            model="gpt-5-nano",
            tools=[{"type": "web_search"}],
            reasoning={"effort": "low"},
            input=prompt,
            max_output_tokens=20000,
        )

        with open(filename, "w", encoding="utf-8") as f:
            f.write(response.output_text)
    
    except Exception:
        print('probably hit rate limit of openai api')
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('')
        
        await asyncio.sleep(90)  # 90 seconds to reset TPM rate limit

    return filename


async def all_agents_fetch_news(
    config, write_to_path: str = None
) -> (str, Dict[str, List[str]]):
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
        second_iteration_tasks.append(
            single_agent_news(
                agent_name, topics, iteration=2, previous_news=previous_news
            )
        )

    second_iteration_files = await asyncio.gather(*second_iteration_tasks)

    for agent_name, file2 in zip(agent_names, second_iteration_files):
        agent_files[agent_name].append(file2)

    # Concatenate all agent files into one raw news file
    all_results = []
    for files in agent_files.values():
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                all_results.append(f.read())

    raw_news = "\n\n".join(all_results)

    if write_to_path:
        with open(write_to_path, "w", encoding="utf-8") as f:
            f.write(raw_news)

    return raw_news, agent_files


async def split_raw_news(raw_news: str, write_to_path: str = None):
    """
    Split RAW news text into structured JSON items, extracting:
    - full_text (everything between <NEWS_ITEM> and </NEWS_ITEM>)
    - priority (integer from <PRIORITY> tag)
    """

    # Regex for each news block
    news_pattern = re.compile(
        r"<NEWS_ITEM>(.*?)</NEWS_ITEM>", re.DOTALL | re.IGNORECASE
    )
    news_blocks = news_pattern.findall(raw_news)

    items = []
    for block in news_blocks:
        block_clean = block.strip()
        prio_match = re.search(
            r"<PRIORITY>(\d+)</PRIORITY>", block_clean, re.IGNORECASE
        )
        priority = int(prio_match.group(1)) if prio_match else 0
        # Remove priority tag from full_text
        full_text_clean = re.sub(
            r"<PRIORITY>\d+</PRIORITY>", "", block_clean, flags=re.IGNORECASE
        ).strip()

        items.append({"full_text": full_text_clean, "priority": priority})

    # with open(chunks_json_path, "w", encoding="utf-8") as f:
    #     json.dump({"items": items}, f, ensure_ascii=False, indent=2)

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
    for item in news_items["items"]:
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


async def find_closest_past_news(news_embed, top_n=3):
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="df_news_embeddings")
    results = collection.query(
        query_embeddings=[news_embed], n_results=3, include=["documents"]
    )

    return results["documents"][:top_n]


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
    past_news_formatted = (
        "\n\n---\n\n".join(
            [f"Past News {i + 1}:\n{news}" for i, news in enumerate(past_news_list)]
        )
        if past_news_list
        else "No past news found"
    )

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
            {"role": "user", "content": prompt},
        ],
        response_format=DuplicateCheckResponse,
    )

    result = response.choices[0].message.parsed

    # Return True if NOT duplicate, False if IS duplicate
    return not result.is_duplicate, news_embedding


async def remove_duplicates(top_news, write_to_path=None):
    """
    Remove duplicate news items by checking each against past news.
    Returns fresh (non-duplicate) news and their embeddings.
    Writes fresh news to fresh_news.txt in raw format compatible with curate_news.
    """
    tasks = []
    for news_item in top_news:
        news_text = news_item["full_text"]
        tasks.append(llm_check_duplicate(news_text))

    results = await asyncio.gather(*tasks)

    fresh_news = []
    embeddings = []

    i = 0  # add 11 news items maximum
    for news_item, (is_not_duplicate, news_embedding) in zip(top_news, results):
        if is_not_duplicate:  # filter duplicates
            i += 1
            fresh_news.append(news_item)
            embeddings.append(news_embedding)
            if i > 10:
                break

    fresh_news_raw = "\n\n".join([f"{item['full_text']}\n\n" for item in fresh_news])

    if write_to_path:
        with open(write_to_path, "w", encoding="utf-8") as f:
            f.write(fresh_news_raw)

    return fresh_news, embeddings


async def write_text_vectors(fresh_news, embeddings):
    """
    Write fresh news items and their embeddings to ChromaDB.
    """
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="df_news_embeddings")

    # Prepare data for ChromaDB
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    ids = [f"news_{i}_{current_timestamp}" for i in range(len(fresh_news))]
    documents = [item["full_text"] for item in fresh_news]
    metadatas = [{"added_date": current_date} for item in fresh_news]

    # Add to ChromaDB if any fresh news
    if len(fresh_news) > 0:
        collection.add(
            ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
        )

    print(f"Added {len(fresh_news)} news items to ChromaDB with date {current_date}")


async def curate_news(fresh_news: str, write_to_path=None) -> str:
    """
    Process raw news data to create a curated list of top 11 impactful news items.
    """

    # Build agent topics context

    prompt = f"""
    You are given raw news/events data collected by multiple agents. Process this data according to these requirements:
    
    1. Delete any duplicate news items covering the same event. Leave only one such news item.
    2. Visit the links provided to verify the date, link and actuality of the news item. If date not correct, fix it. 
    3. Pay special attention when the date in the link string contradicts the date in the news item.
    4. If news item doesn't have a date or a link, try to find it. 
    5. If the actual link can not be found, discard the news item.
    
    
    OUTPUT FORMAT:
    6. Format each news item with this exact structure so that they were uniform:
       
       ## [Headline]
       [Several sentence summary copy-pasted from the given news.]    
       Date:[publication date or most recent event date. Date must be only here]
       Link:[URL]

    7. No '-' or ":" or other special characters in the headline
    8. Summary should be a single paragraph, without any bullet points or lists. It should not repeat what's already in the headline.
    9. The headline should be short. Preffered format is "X did Y". 
    10. If the news is taken from somewhere, mentioning the source in the headline or summary is redundant because the link exists.
       
       ---
    
    RAW NEWS DATA:
    {fresh_news}
    
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

    if write_to_path:
        with open(write_to_path, "w", encoding="utf-8") as f:
            f.write(curated_news)

    return curated_news


async def html_news(semi_formatted_news: str, write_to_path=None) -> str:
    """
    Convert semi-formatted news text to a well-formatted HTML file.

    Args:
        semi_formatted_news: The text content with news items
        output_path: Path to save the HTML file

    Returns:
        Path to the generated HTML file
    """

    prompt = f"""
    Convert the following news items into a well-formatted HTML document. 
    
    Requirements:
    1. Create a clean, readable HTML page with proper structure (<!DOCTYPE html>, <html>, <head>, <body>)
    2. Include CSS styling for:
       - Clear separation between news items
       - Proper styling for headlines, dates, links, and content
    3. Each news item should be clearly separated and easy to read
    4. Preserve ALL content including headlines, summaries, dates, and links
    5. Make links clickable
    6. Add a title to the page like "News Digest - [Current Date]"
    7. Use semantic HTML elements
    
    NEWS CONTENT:
    {semi_formatted_news}
    
    Output only the complete HTML code, nothing else.
    """

    client = AsyncOpenAI()

    response = await client.chat.completions.create(
        model="gpt-5-mini",
        reasoning_effort="low",
        messages=[
            {
                "role": "system",
                "content": "You are an expert HTML/CSS developer who creates clean, well-formatted web pages.",
            },
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=15000,
    )

    html_content = response.choices[0].message.content

    # Remove markdown code fences if present
    html_content = re.sub(r"^```html\n", "", html_content)
    html_content = re.sub(r"\n```$", "", html_content)
    html_content = html_content.strip()

    if write_to_path:
        with open(write_to_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    return html_content


def print_section(message):
    """Print a message wrapped with separator lines."""
    print("\n" + "=" * 50)
    print(message)
    print("=" * 50 + "\n")


async def main():
    global client

    # Initialize OpenAI client based on --openai_api_key argument
    if args.openai_api_key == "False":
        # If argument's value is False, then OpenAI API won't be used
        client = None
        use_openai = False
    elif args.openai_api_key:
        # Use provided API key
        client = AsyncOpenAI(api_key=args.openai_api_key)
        use_openai = True
    elif os.environ.get("OPENAI_API_KEY"):
        # Fall back to .env file
        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        use_openai = True
    else:
        client = None
        use_openai = False

    if args.use_telegram_api:
        print_section("Fetching Telegram context...")
        telegram_context_file = await get_telegram_context(
            config,
            write_to_path="telegram_context.txt",
            api_hash=args.api_hash,
            api_id=args.api_id,
        )

        print_section("Extracting news from Telegram messages...")
        telegram_news_raw = await extract_telegram_news(config, telegram_context_file)

        # split telegram news
        print_section("Splitting RAW Telegram news into structured chunks...")
        telegram_chunks = await split_raw_news(
            raw_news=telegram_news_raw,
            write_to_path="news_raw_telegram_separate_items.json",
        )
        telegram_list_of_news = telegram_chunks.output_parsed

        print_section("Selecting up to 40 Telegram news with highest priority")
        top_telegram_news = await shuffle_sort_news(telegram_list_of_news, 40)

        print_section("projecting Telegram news items into embeddings...")
        fresh_telegram_news, telegram_embeddings = await remove_duplicates(
            top_telegram_news, write_to_path="fresh_telegram_news.txt"
        )

        print_section("Writing Telegram news embeddings to ChromaDB...")
        await write_text_vectors(fresh_telegram_news, telegram_embeddings)

        # write fresh_telegram_news as a file
        fresh_telegram_news = "\n\n".join(
            [f"{item['full_text']}\n\n" for item in fresh_telegram_news]
        )
        # with open("fresh_telegram_news.txt", "w", encoding="utf-8") as f:
        #     f.write(fresh_telegram_news)

        print("Telegram news are ready at fresh_telegram_news.txt")

    if use_openai:
        print_section("Fetching news according to topics in the config")
        raw_news, _ = await all_agents_fetch_news(config)

        print_section("Splitting RAW news into structured chunks...")
        raw_chunks = await split_raw_news(
            raw_news=raw_news, write_to_path="news_raw_separate_items.json"
        )
        list_of_news = raw_chunks.output_parsed

        print_section("Selecting 60 news with highest priority")
        top_news = await shuffle_sort_news(list_of_news, 60)

        print_section("projecting news items into embeddings...")
        fresh_news, embeddings = await remove_duplicates(
            top_news, write_to_path="fresh_news.txt"
        )

        print_section("Writing embeddings to ChromaDB...")
        await write_text_vectors(fresh_news, embeddings)

        print_section("Curating news...")
        curated_news = await curate_news(fresh_news, write_to_path="web_news.txt")

        print("News from web are ready at web_news.txt")

    if args.use_telegram_api and use_openai:
        print_section("Combining curated news with Telegram news...")

        combined_news = curated_news + "\n\n" + fresh_telegram_news

        with open("web_telegram_news.txt", "w", encoding="utf-8") as f:
            f.write(combined_news)

        print("Combined news from web and Telegram are ready at web_telegram_news.txt")

    # Generate HTML output based on what's available
    print_section("Generating HTML news page...")
    match (args.use_telegram_api, use_openai):
        case (True, True):
            await html_news(combined_news, write_to_path="news.html")
        case (False, True):
            await html_news(curated_news, write_to_path="news.html")
        case (True, False):
            await html_news(fresh_telegram_news, write_to_path="news.html")

    print("\nfinished!")


if __name__ == "__main__":
    args = parse_args()

    config = load_config(args.config)

    asyncio.run(main())
