from __future__ import annotations

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="KPSS Adaptive RAG", page_icon="📚", layout="wide")
st.title("📚 KPSS Adaptif Soru Öneri Sistemi")
st.caption("Çözüm geçmişi → öğrenci profili → zayıf konu + seviye bandı → vektör retrieval → ranking")

with st.sidebar:
    st.header("Ayarlar")
    user_id = st.text_input("Kullanıcı ID", value="u_001")
    target_lesson = st.selectbox("Ders filtresi", ["", "Vatandaşlık", "Tarih", "Türkçe", "Matematik"])
    rebuild = st.checkbox("İndeksi yeniden oluştur", value=False)
    st.info("Önce terminalde `uvicorn main_api:app --reload` çalışmalı.")

if "recommended" not in st.session_state:
    st.session_state.recommended = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None

col1, col2 = st.columns([2, 1])

with col1:
    if st.button("Seviyeme uygun soru getir", type="primary"):
        payload = {
            "user_id": user_id,
            "target_lesson": target_lesson or None,
            "rebuild_index": rebuild,
        }
        try:
            response = requests.post(f"{API_BASE}/kpss/recommend-question", json=payload, timeout=120)
            response.raise_for_status()
            st.session_state.recommended = response.json()
            st.session_state.last_result = None
        except Exception as exc:
            st.error(f"Soru getirilemedi: {exc}")

    rec = st.session_state.recommended
    if rec:
        q = rec["selected_question"]
        st.subheader(f"{q['lesson']} / {q['topic']}")
        st.write(f"**Alt konu:** {q.get('subtopic') or '-'}")
        st.write(f"**Zorluk:** {q['difficulty']:.2f}")
        if q.get("quality_warning"):
            st.warning(f"Veri kalite uyarısı: {q['quality_warning']}")
        st.markdown(f"### {q['question_text']}")
        user_answer = st.text_area(
            "Cevabın",
            key=q["question_id"],
            height=140,
            placeholder="Cevabını kısa ve açık şekilde yaz.",
        )
        response_time = st.number_input("Cevaplama süresi (sn)", min_value=0.0, value=45.0)
        if st.button("Cevabı gönder"):
            payload = {
                "user_id": user_id,
                "question_id": q["question_id"],
                "question_text": q["question_text"],
                "reference_answer": q["reference_answer"],
                "student_answer": user_answer,
                "grading_context": q.get("grading_context") or q["reference_answer"],
                "accepted_aliases": q.get("accepted_aliases", []),
                "key_concepts": q.get("key_concepts", []),
                "wrong_if_mentions": q.get("wrong_if_mentions", []),
                "partial_credit_rules": q.get("partial_credit_rules", []),
                "lesson": q["lesson"],
                "topic": q["topic"],
                "difficulty": q["difficulty"],
                "response_time": response_time,
                "persist": True,
            }
            try:
                response = requests.post(f"{API_BASE}/kpss/submit-open-answer", json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                st.session_state.last_result = result
                evaluation = result["evaluation"]
                st.success(f"Değerlendirme: {evaluation['label']} | Skor: {evaluation['score']:.2f}")
                st.write(f"**Referans cevap:** {evaluation['expected_answer']}")
                st.write(evaluation["feedback"])
                st.write(f"**Güncellenmiş seviye:** {result['updated_user_level_preview']:.2f}")
            except Exception as exc:
                st.error(f"Cevap değerlendirilemedi: {exc}")

with col2:
    st.subheader("Neden bu soru önerildi?")
    rec = st.session_state.recommended
    if rec:
        st.write(rec["selection_reason"])
        st.json(rec["score_breakdown"])
        st.write("**Zayıf konu:**", rec.get("weak_topic"))
        st.write("**Öğrenci seviyesi:**", f"{rec['user_level']:.2f}")
        if rec.get("profile_summary"):
            st.divider()
            st.write("**Profil özeti**")
            profile = rec["profile_summary"]
            st.write("Accuracy:", f"{profile['accuracy']:.2f}")
            st.write("Recent accuracy:", f"{profile['recent_accuracy']:.2f}")
            st.write("Ortalama çözülen zorluk:", f"{profile['avg_solved_difficulty']:.2f}")
            st.write("Zayıf konular:", profile["weak_topics"][:5])
    else:
        st.write("Henüz soru seçilmedi.")

    if st.session_state.last_result:
        st.divider()
        st.subheader("Güncellenmiş kullanıcı profili")
        st.json(st.session_state.last_result["updated_profile"])
