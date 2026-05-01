from __future__ import annotations

import requests

API_URL = "http://127.0.0.1:8000/ask"


def main() -> None:
    question = input("PDF hakkında soru sorun: ").strip()
    response = requests.post(API_URL, json={"question": question}, timeout=120)
    response.raise_for_status()
    data = response.json()

    print("\nCevap:")
    print(data["answer"])
    print("\nKaynaklar:")
    for src in data["sources"]:
        print(f"- {src['source']} | sayfa={src['page']} | {src['content_preview'][:160]}...")


if __name__ == "__main__":
    main()
