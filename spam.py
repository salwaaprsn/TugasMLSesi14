import streamlit as st
import pandas as pd
import pickle
from sklearn.base import ClassifierMixin

# Load model dan vectorizer
with open("model.pkl", "rb") as f:
    model: ClassifierMixin = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.set_page_config(page_title="Deteksi Spam", page_icon="📨")

# Inisialisasi session_state untuk riwayat
if "history" not in st.session_state:
    st.session_state.history = []

st.title("📨 Deteksi Pesan Spam")

tab1, tab2 = st.tabs(["🧍 Deteksi Individu", "📂 Deteksi Massal (CSV)"])

with tab1:
    st.markdown("Masukkan pesan teks di bawah ini untuk mendeteksi apakah termasuk spam atau bukan.")

    user_input = st.text_area("✏️ Masukkan pesan:")

    with st.expander("⚙️ Pengaturan Lanjutan"):
        threshold = st.slider("Threshold untuk klasifikasi sebagai SPAM (%)", min_value=0, max_value=100, value=50)

    if st.button("🚀 Deteksi"):
        if user_input.strip() == "":
            st.warning("Silakan masukkan pesan terlebih dahulu.")
        else:
            input_vect = vectorizer.transform([user_input])

            try:
                proba = model.predict_proba(input_vect)[0]
                spam_proba = round(proba[1] * 100, 2)
                ham_proba = round(proba[0] * 100, 2)

                prediction = "spam" if spam_proba >= threshold else "ham"
            except:
                prediction = model.predict(input_vect)[0]
                spam_proba = ham_proba = "-"

            if prediction.lower() == "spam":
                st.error("📛 **Hasil Deteksi: SPAM**")
            else:
                st.success("✅ **Hasil Deteksi: Bukan Spam (HAM)**")

            st.session_state.history.append({
                "Pesan": user_input,
                "Hasil": prediction,
                "Spam (%)": spam_proba,
                "Ham (%)": ham_proba
            })

    if st.session_state.history:
        st.markdown("### 📜 Riwayat Deteksi Sebelumnya")
        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history, use_container_width=True)

        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("🗑️ Hapus Riwayat"):
                st.session_state.history.clear()
                st.success("Riwayat berhasil dihapus.")

        with col2:
            st.download_button("⬇️ Unduh Riwayat (CSV)", df_history.to_csv(index=False), file_name="riwayat_deteksi.csv", mime="text/csv")

        st.markdown("### 📊 Ringkasan Visual Deteksi")
        df_count = df_history["Hasil"].value_counts().reset_index()
        df_count.columns = ["Kategori", "Jumlah"]
        st.bar_chart(df_count.set_index("Kategori"))

with tab2:
    st.markdown("Unggah file `.csv` dengan kolom berisi pesan teks. Contoh header: `pesan`")

    uploaded_file = st.file_uploader("📂 Unggah file CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "pesan" not in df.columns:
                st.error("File harus memiliki kolom bernama 'pesan'")
            else:
                vect_texts = vectorizer.transform(df["pesan"])
                try:
                    probas = model.predict_proba(vect_texts)
                    df["Spam (%)"] = (probas[:, 1] * 100).round(2)
                    df["Ham (%)"] = (probas[:, 0] * 100).round(2)
                    df["Hasil"] = ["spam" if p >= threshold else "ham" for p in df["Spam (%)"]]
                except:
                    predictions = model.predict(vect_texts)
                    df["Spam (%)"] = "-"
                    df["Ham (%)"] = "-"
                    df["Hasil"] = predictions

                st.success("✅ Deteksi selesai!")
                st.dataframe(df[["pesan", "Hasil", "Spam (%)", "Ham (%)"]], use_container_width=True)

                st.download_button("⬇️ Unduh Hasil Deteksi", df.to_csv(index=False), file_name="hasil_massal.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")