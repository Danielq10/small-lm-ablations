import json
import urllib.request


def get_wolne_lektury_txt_urls() -> list[str]:
    url = "https://wolnelektury.pl/api/books/"
    with urllib.request.urlopen(url) as response:
        books = json.loads(response.read().decode("utf-8"))

    return [
        f"https://wolnelektury.pl/media/book/txt/{book['slug']}.txt" for book in books
    ]


if __name__ == "__main__":
    txt_urls = get_wolne_lektury_txt_urls()
    print(f"Total books found: {len(txt_urls)}")

    with open("data/book_urls.json", "w", encoding="utf-8") as f:
        json.dump(txt_urls, f, ensure_ascii=False, indent=4)
