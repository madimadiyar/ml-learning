import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/bart-large-xsum"
news_url = "https://www.cnn.com/2026/10/27/politics/jared-moskowitz-florida-pro-israel-democrat"
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

def summarize(text: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=False).input_ids
    outputs = model.generate(inputs, max_new_tokens=60, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    news = fetch_html(news_url)
    result = extract_text(news)
    short = result[:3000]
    print(summarize(short))


if __name__ == '__main__':
    main()
