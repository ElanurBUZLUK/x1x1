# KPSS Adaptive RAG Question Recommender

Bu proje, temel `rag_project` yapısının üzerine kurulan **RAG destekli Spotify-benzeri KPSS soru öneri sistemi**dir.

Temel fikir:

```text
Spotify: dinleme geçmişi + şarkı özellikleri → kişiye uygun şarkı önerisi
Bu proje: çözüm geçmişi + soru özellikleri → seviyeye uygun KPSS sorusu önerisi
```

## Proje ne yapar?

1. KPSS soru bankasını JSON formatında okur.
2. Her soruyu tek bir retrieval dokümanı olarak Chroma vektör veritabanına kaydeder.
3. Öğrenci çözüm geçmişinden seviye ve konu bazlı zayıflık profili çıkarır.
4. Chroma'dan aday sorular getirir.
5. Adayları Spotify-benzeri skorla sıralar:
   - konu uygunluğu
   - zorluk uygunluğu
   - zayıf konu eşleşmesi
   - semantik benzerlik
   - daha önce çözülmemiş olma
6. Seçilen soru için LLM ile açıklama / geri bildirim üretir.
7. Öğrencinin açık uçlu cevabını referans cevaba göre puanlar ve profili günceller.

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate      # Windows

pip install -r requirements.txt
ollama pull gemma3:1b
```

## 1. KPSS soru vektör veritabanını oluştur

```bash
python -m src.question_ingest
```

veya öneri demosuyla birlikte yeniden oluştur:

```bash
python main_recommend.py --rebuild --lesson Vatandaşlık
```

## 2. Terminalden soru önerisi al

```bash
python main_recommend.py --lesson Vatandaşlık
```

## 3. API servisini çalıştır

```bash
uvicorn main_api:app --reload
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

Önemli endpointler:

```text
POST /kpss/rebuild-index
POST /kpss/recommend-question
POST /kpss/evaluate-open-answer
POST /kpss/submit-open-answer
```

## 4. Streamlit arayüzünü çalıştır

Ayrı terminalde API çalışırken:

```bash
streamlit run streamlit_app.py
```

## Veri dosyaları

```text
data/questions/kpss_questions_sample.json
```

Her soru şu mantıkla tutulur:

```json
{
  "question_id": "kpss_vat_0001",
  "lesson": "Vatandaşlık",
  "topic": "Anayasa Hukuku",
  "subtopic": "Temel Hak ve Ödevler",
  "difficulty": 0.42,
  "question_text": "...",
  "reference_answer": "...",
  "accepted_aliases": ["..."],
  "key_concepts": ["..."],
  "grading_context": "...",
  "wrong_if_mentions": ["..."],
  "partial_credit_rules": ["..."],
  "explanation": "...",
  "tags": ["anayasa", "temel haklar"]
}
```

Öğrenci geçmişi:

```text
data/users/sample_user_history.json
```

## Ana dosyalar

```text
src/question_ingest.py        # soru bankasını Chroma'ya aktarır
src/user_profile.py           # öğrenci profilini çıkarır
src/level_estimator.py        # seviye etiketi / zorluk bandı
src/weakness_detector.py      # zayıf konu seçimi
src/adaptive_retriever.py     # aday retrieval + öneri
src/ranking.py                # Spotify-benzeri skor formülü
src/explanation_generator.py  # LLM ile açıklama/geri bildirim
src/open_answer_evaluator.py  # açık uçlu cevap puanlama
main_recommend.py             # terminal demosu
main_api.py                   # FastAPI servisi
streamlit_app.py              # demo arayüz
```

## Scoring mantığı

```text
final_score =
    0.30 * topic_match
  + 0.25 * difficulty_match
  + 0.20 * weakness_match
  + 0.15 * semantic_similarity
  + 0.10 * novelty_score
```

## Notlar

- LLM yeni KPSS sorusu üretmez; mevcut soru bankasından seçilen soruyu açıklar.
- Projede şıklı cevap akışı yoktur; cevaplar serbest metin olarak değerlendirilir.
- Bu yaklaşım sınav sorularında doğruluk ve kontrol için daha güvenlidir.
- İlk MVP content-based recommendation + RAG explanation mantığıyla kurulmuştur.
- Daha ileri fazlarda collaborative filtering, IRT, Bayesian Knowledge Tracing veya bandit yaklaşımı eklenebilir.
