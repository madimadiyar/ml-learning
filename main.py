import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "google-t5/t5-small"
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36)'}


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    paragraphs = soup.find_all('p')
    items =[]
    for p in paragraphs:
        items.append(p.text)

    return "\n".join(items)


def fetch_html(url: str) -> str:
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text

def summarize(html: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    inputs = tokenizer(html, return_tensors="pt", padding=True, truncation=True).input_ids
    outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    news = fetch_html(
        "https://www.bbc.com/news/articles/ck87zlpzny15o")
    result = extract_text(news)
    short = result[:3000]
    print(summarize(short))


if __name__ == '__main__':
    main()
