import asyncio
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md

import yaml


def url_to_filename(url):
    """
    Converts a URL to a valid filename for saving Markdown content.

    Behavior
    --------
    - Converts the URL path to a filename, replacing slashes with underscores.
    - Removes any characters that are not alphanumeric, underscores, hyphens, or periods.
    - Ensures the filename ends with `.md`.

    Parameters
    ----------
    - url : `str`
        The URL to convert.

    Returns
    -------
    - filename : `str`
        The converted filename.
    """
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


async def process_page(
    session: httpx.AsyncClient, url: str, domain_dir: Path, visited_urls: set
) -> list[str]:
    """
    Process a single page, extract its main content, and save it as Markdown.

    Behavior
    --------
    This function fetches the content of a given URL, extracts the main content,
    converts it to Markdown, and saves it to a file. It also collects new links.

    Parameters
    ----------
    - session : `httpx.AsyncClient`
        The HTTP session to use for requests.
    - url : `str`
        The URL of the page to process.
    - domain_dir : `Path`
        The directory to save the Markdown files.
    - visited_urls : `set`
        A set of URLs that have already been visited.

    Returns
    -------
    - new_links: `list[str]`
        A list of newly discovered links.
    """
    if url in visited_urls:
        return []

    try:
        response = await session.get(url, timeout=10, follow_redirects=True)

        visited_urls.add(url)
        final_url = str(response.url)
        visited_urls.add(final_url)

        response.raise_for_status()
        print(f"  -> Processing: {final_url} (Status: {response.status_code})")

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

        new_links = []
        base_domain = urlparse(final_url).netloc
        for a_tag in soup.find_all("a", href=True):
            link = a_tag["href"]  # type: ignore
            absolute_link = urljoin(final_url, link)  # type: ignore
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


async def worker(
    name,
    queue: asyncio.Queue,
    session: httpx.AsyncClient,
    domain_dir: Path,
    visited_urls: set,
    semaphore: asyncio.Semaphore,
    max_depth: int,
):
    """
    Worker function that processes URLs from the queue.

    Behavior
    --------
    This function continuously pulls URLs from the queue and processes them using the provided
    HTTP session. It respects the maximum depth and uses a semaphore to limit concurrency.

    Parameters
    ----------
    - name : `str`
        The name of the worker.
    - queue : `asyncio.Queue`
        The queue of URLs to process.
    - session : `httpx.AsyncClient`
        The HTTP session to use for requests.
    - domain_dir : `Path`
        The directory to save the Markdown files.
    - visited_urls : `set`
        A set of URLs that have already been visited.
    - semaphore : `asyncio.Semaphore`
        A semaphore to limit concurrency.
    - max_depth : `int`
        The maximum depth to crawl.
    """
    while True:
        current_url, current_depth = await queue.get()

        if current_depth > max_depth:
            queue.task_done()
            continue

        async with semaphore:
            newly_found_links = await process_page(
                session, current_url, domain_dir, visited_urls
            )

            if current_depth < max_depth:
                for link in newly_found_links:
                    if link not in visited_urls:
                        await queue.put((link, current_depth + 1))

        queue.task_done()


async def crawl_site_async(
    seed_url: str, output_dir: Path, max_depth=2, concurrency=20
):
    """
    Crawl a website asynchronously and save its content as Markdown files.

    Behavior
    --------
    This function initiates the crawling process for a single website, starting from the seed URL.
    It manages the asynchronous tasks and ensures that the crawling adheres to the specified depth
    and concurrency limits.

    Parameters
    ----------
    - seed_url : `str`
        The starting URL for the crawl.
    - output_dir : `Path`
        The directory where the Markdown files will be saved.
    - max_depth : `int`. Optional, by default 2
        The maximum depth to crawl.
    - concurrency : `int`. Optional, by default 20
        The maximum number of concurrent requests.
    """
    queue = asyncio.Queue()
    semaphore = asyncio.Semaphore(concurrency)
    visited_urls = set()

    base_domain = urlparse(seed_url).netloc
    domain_dir = output_dir / base_domain.replace(".", "_")
    domain_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[*] Starting async crawl on: {base_domain} with concurrency: {concurrency}, depth: {max_depth}"
    )

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

        await queue.join()

        for w in workers:
            w.cancel()

        await asyncio.gather(*workers, return_exceptions=True)

    print(f"\n[*] Crawl finished for {base_domain}.")
    print(f"[*] Found and processed {len(visited_urls)} unique pages.")


async def main():
    with open(Path(__file__).parent / "input/links.yaml") as file:
        urls = yaml.safe_load(file)
    URLS_TO_CRAWL = []
    for web in urls.keys():
        URLS_TO_CRAWL += urls[web]["main_url"]

    MAIN_OUTPUT_DIR = Path(__file__).parent / "output"
    MAX_DEPTH = 1
    CONCURRENCY_LIMIT = 25  # Number of parallel requests

    start_time = time.time()

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
    asyncio.run(main())
