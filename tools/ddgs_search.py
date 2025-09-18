from ddgs import DDGS

def ddgs_search(query: str, max_results: int = 5):
    results = DDGS().text(query, max_results=max_results)
    return [{'href': r.get('href', ''), 'body': r.get('body', '')} for r in results]

# print(ddgs_search("Python programmling language", max_results=3))