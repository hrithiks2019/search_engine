'''Example:
    python search.py 'ruby go' https://jorin.me
'''
import re
import argparse
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen
from urllib.error import HTTPError
from collections import Counter, defaultdict
from math import log10
from bs4 import BeautifulSoup
import numpy as np

tpx,tdx,stop_words = 0.05,0.04,['a', 'also', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'do',
					'for', 'have', 'is', 'in', 'it', 'of', 'or', 'see', 'so',
					'that', 'the', 'this', 'to', 'we']
  
def get_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'query',
        type=str,
        help='Search query string can contain multiple words'
    )
    parser.add_argument(
        'url',
        type=str,
        nargs='+',
        help='At least one seed url for the crawler to start from'
    )
    return parser.parse_args()


def crawl(urls, _frontier={}, _bases=None):
    '''
    Takes a list of urls as argument and crawls them recursivly until
    no new url can be found.

    Returns a sorted list of tuples (url, content, links).
    `links` is a list of urls.
    '''
    if not _bases:
        _bases = [urlparse(u).netloc for u in urls]
    for url in [u.rstrip('/') for u in urls]:
        if url in _frontier:
            continue
        try:
            response = download(url)
        except HTTPError as e:
            print(e, url)
            continue

        page = parse(response, url, _bases)
        print('crawled %s with %s links' % (url, len(page[2])))
        _frontier[url] = page
        crawl(page[2], _frontier, _bases)
    return sorted(_frontier.values())


def download(url):
    return urlopen(url)


def parse(html, url, bases):
    soup = BeautifulSoup(html, 'lxml')

    content = soup.body.get_text().strip()

    links = [urljoin(url, l.get('href')) for l in soup.findAll('a')]
    links = [l for l in links if urlparse(l).netloc in bases]
    return url, content, links


def page_rank(pages):
    N = len(pages)
    transition_matrix = create_transition_matrix(pages)
    ranks_in_steps = [[1 / N] * N]
    while True:
        possibilities = ranks_in_steps[-1] * transition_matrix
        delta = get_delta(possibilities, ranks_in_steps[-1])
        ranks_in_steps.append(np.squeeze(np.asarray(possibilities)))
        if delta <= tdx:
            return ranks_in_steps


def create_transition_matrix(pages):
    links = get_links(pages)
    urls = get_urls(pages)
    N = len(pages)
    m = np.matrix([[weight_link(N, u, l) for u in urls] for l in links])
    return teleport(N, m)


def weight_link(N, url, links):
    if not links:
        return 1 / N
    if url in links:
        return 1 / len(links)
    else:
        return 0


def teleport(N, m):
    return m * (1 - tpx) + tpx / N


def get_delta(a, b):
    return np.abs(a - b).sum()


def get_urls(pages):
    return [url for url, content, links in pages]


def get_links(pages):
    return [links for url, content, links in pages]


def best_rank(ranks, pages):
    return dict(zip(get_urls(pages), ranks[-1]))


# Index

def create_index(pages):
    index = defaultdict(list)
    for url, content, links in pages:
        counts = count_terms(content)
        for term, count in counts.items():
            index[term].append((url, count))
    return index


def count_terms(content):
    return Counter(get_terms(content))


normalize = re.compile('[^a-z0-9]+')


def get_terms(s):
    normalized = [normalize.sub('', t.lower()) for t in s.split()]
    return [t for t in normalized if t not in stop_words]


def weight_index(index, N):
    weighted_index = defaultdict(list)
    for term, docs in index.items():
        df = len(docs)
        for url, count in docs:
            weight = tf_idf(count, N, df)
            weighted_index[term].append((url, weight))
    return weighted_index


def tf_idf(tf, N, df):
    return wtf(tf) * idf(N, df)


def wtf(tf):
    return 1 + log10(tf)


def idf(N, df):
    return log10(N / df)


def normalize_index(index):
    lengths = doc_lengths(index)
    norm_index = defaultdict(list)
    for term, docs in index.items():
        for url, weight in docs:
            norm_index[term].append((url, weight / lengths[url]))
    return norm_index


def doc_lengths(index):
    doc_vectors = defaultdict(list)
    for docs in index.values():
        for url, weight in docs:
            doc_vectors[url].append(weight)
    return {url: np.linalg.norm(doc) for url, doc in doc_vectors.items()}



def cosine_similarity(index, N, query):
    scores = defaultdict(int)
    terms = query.split()
    qw = {t: tf_idf(1, N, len(index[t])) for t in terms if t in index}
    query_len = np.linalg.norm(list(qw.values()))
    for term in qw:
        query_weight = qw[term] / query_len
        for url, weight in index[term]:
            scores[url] += weight * query_weight
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def combined_search(index, N, rank, query):
    scores = cosine_similarity(index, N, query)
    combined = [(doc, score * rank[doc]) for doc, score in scores]
    return sorted(combined, key=lambda x: x[1], reverse=True)


def pcs(index, N, rank, query):
    print('Search results for "%s":' % (query))
    for url, score in combined_search(index, N, rank, query):
        print('%.6f  %s' % (score, url))



args = get_args()
pages = crawl(args.url)
ranks = page_rank(pages)
rank = best_rank(ranks, pages)
N = len(pages)
index = create_index(pages)
weighted_index = weight_index(index, N)
norm_index = normalize_index(weighted_index)
print('No.of.pages:', len(pages))
print('PageRank:', len(ranks))
pcs(norm_index, N, rank, args.query)
