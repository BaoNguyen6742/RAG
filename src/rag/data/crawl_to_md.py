import asyncio
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx  # The async-compatible requests library
from bs4 import BeautifulSoup
from markdownify import markdownify as md

import yaml


def url_to_filename(url):
    """Converts a URL into a safe filename string."""
    parsed_url = urlparse(url)
    path_str = parsed_url.path.lstrip("/")
    if not path_str:
        filename = "index"
    else:
        filename = path_str.replace("/", "_")
    filename = re.sub(r"[^a-zA-Z0-9_.-]", "", filename)
    if not filename.endswith(".md"):
        filename += ".md"
    return filename


async def process_page(session, url, domain_dir, visited_urls):
    """
    Fetches a single page, saves it, and finds new links.
    Returns a list of newly discovered links.
    """
    # The check for visited URLs is now the single source of truth.
    if url in visited_urls:
        return []

    try:
        # The 'follow_redirects=True' is important.
        response = await session.get(url, timeout=10, follow_redirects=True)

        # Mark both the original URL and the final URL (after redirects) as visited.
        # This is done *after* a successful request.
        visited_urls.add(url)
        final_url = str(response.url)
        visited_urls.add(final_url)

        response.raise_for_status()
        print(f"  -> Processing: {final_url} (Status: {response.status_code})")

        # --- Save content to Markdown ---
        soup = BeautifulSoup(response.text, "html.parser")
        main_content = (
            soup.find("div", role="main") or soup.find("main") or soup.find("article")
        )
        if main_content:
            markdown_content = md(str(main_content), heading_style="ATX")
            filepath = domain_dir / url_to_filename(final_url)
            filepath.write_text(markdown_content, encoding="utf-8")
        else:
            print(f"    -> Warning: Main content not found for {final_url}")

        # --- Find and return new links ---
        new_links = []
        base_domain = urlparse(final_url).netloc
        for a_tag in soup.find_all("a", href=True):
            link = a_tag["href"]  # type: ignore
            absolute_link = urljoin(final_url, link) # type: ignore
            link_domain = urlparse(absolute_link).netloc

            if (
                link_domain == base_domain
                and "#" not in absolute_link
                and not absolute_link.endswith((".pdf", ".jpg", ".zip", ".png"))
            ):
                new_links.append(absolute_link)
        return new_links

    except httpx.RequestError as e:
        print(f"  -> HTTP Error for {e.request.url}: {type(e).__name__}")
        return []
    except Exception as e:
        print(f"  -> An error occurred processing {url}: {e}")
        return []


async def worker(name, queue, session, domain_dir, visited_urls, semaphore, max_depth):
    """A worker task that pulls URLs from the queue and processes them."""
    while True:
        current_url, current_depth = await queue.get()

        if current_depth > max_depth:
            queue.task_done()
            continue

        async with semaphore:  # Acquire the semaphore before processing
            newly_found_links = await process_page(
                session, current_url, domain_dir, visited_urls
            )

            if current_depth < max_depth:
                for link in newly_found_links:
                    # Now we can simply check if the link is in the global visited set.
                    # If not, add it to the queue for a future worker to process.
                    if link not in visited_urls:
                        await queue.put((link, current_depth + 1))

        queue.task_done()


async def crawl_site_async(seed_url, output_dir: Path, max_depth=2, concurrency=20):
    """
    Sets up and runs the asynchronous crawl for a single website.
    """
    queue = asyncio.Queue()
    semaphore = asyncio.Semaphore(concurrency)
    visited_urls = set()  # This set is now only written to from within process_page

    base_domain = urlparse(seed_url).netloc
    domain_dir = output_dir / base_domain.replace(".", "_")
    domain_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[*] Starting async crawl on: {base_domain} with concurrency: {concurrency}, depth: {max_depth}"
    )

    # --- THE CRITICAL FIX ---
    # Only put the seed_url in the queue. DO NOT add it to visited_urls here.
    await queue.put((seed_url, 0))

    async with httpx.AsyncClient() as session:
        workers = [
            asyncio.create_task(
                worker(
                    f"w{i}",
                    queue,
                    session,
                    domain_dir,
                    visited_urls,
                    semaphore,
                    max_depth,
                )
            )
            for i in range(concurrency)
        ]

        await queue.join()  # Wait for all tasks in the queue to be processed

        for w in workers:
            w.cancel()

        await asyncio.gather(*workers, return_exceptions=True)

    print(f"\n[*] Crawl finished for {base_domain}.")
    print(f"[*] Found and processed {len(visited_urls)} unique pages.")
    

async def main():
    """The main entry point for the async script."""
    
    with open(Path(__file__).parent / "input/links.yaml") as file:
        urls = yaml.safe_load(file)
    URLS_TO_CRAWL = []
    for web in urls.keys():
        URLS_TO_CRAWL += urls[web]["main_url"]
        
    MAIN_OUTPUT_DIR = Path(__file__).parent / "output"
    MAX_DEPTH = 1
    CONCURRENCY_LIMIT = 25  # Number of parallel requests

    start_time = time.time()

    # Create and run crawl tasks for all sites concurrently
    crawl_tasks = [
        crawl_site_async(
            url, MAIN_OUTPUT_DIR, max_depth=MAX_DEPTH, concurrency=CONCURRENCY_LIMIT
        )
        for url in URLS_TO_CRAWL
    ]
    await asyncio.gather(*crawl_tasks)

    end_time = time.time()
    print(f"\n--- Total execution time: {end_time - start_time:.2f} seconds ---")


if __name__ == "__main__":
    # This is how you run the main async function
    asyncio.run(main())

