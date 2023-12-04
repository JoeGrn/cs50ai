import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    pages = {}
    transition_pages = len(corpus[page])

    if transition_pages:
        # transition probablity
        for page in corpus:
            pages[page] = (1 - damping_factor) / len(corpus)
        # random probability
        for page in corpus[page]:
            pages[page] += damping_factor / transition_pages
    else:
        # random transition as transitions available
        for page in corpus:
            pages[page] = 1 / len(corpus)

    return pages


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages = {}
    for page in corpus:
        pages[page] = 0
        page = random.choice(list(corpus.keys()))

    for i in range(1, n):
        transition = transition_model(corpus, page, damping_factor)
        for page in pages:
            pages[page] = ((i-1) * pages[page] + transition[page])
        
        page = random.choices(list(pages.keys()), list(pages.values()), k=1)[0]

    return pages


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_rank = {}

    # initialise as a random probabilities
    for page in corpus:
        page_rank[page] = 1 / len(corpus)

    looping = True
    while looping:
        looping = False
        for current_page in corpus:
            old_page_rank = page_rank[current_page]
            sum_links = 0

            # sum of the probabilities from other page to current page
            for other_webpage in corpus:
                # if not the same page and other page links to current page
                if current_page != other_webpage and current_page in corpus[other_webpage]:
                    sum_links += page_rank[other_webpage] / len(corpus[other_webpage])

            # update PR value
            new_page_rank = (1 - damping_factor) / len(corpus) + damping_factor * sum_links
            page_rank[current_page] = new_page_rank

            # check if converged, no PR value change greater than 0.001 then exit
            if abs(new_page_rank - old_page_rank) > 0.001:
                looping = True

    return page_rank


if __name__ == "__main__":
    main()
