
import html
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="LSTM Studio | AI Text Generator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. CUSTOM CSS
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --primary: #7357F5;
    --navy: #142448;
    --muted: #7183A4;
    --border: #DDE6F6;
}

.stApp {
    background: #F6F8FD;
    font-family: 'Inter', sans-serif;
    color: var(--navy);
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101A36, #0C1730);
    border-right: 1px solid #26385F;
}

[data-testid="stSidebar"] * {
    color: #F1F5FF;
}

.block-container {
    max-width: 1500px;
    padding: 1.5rem 1.8rem 2.5rem;
}

h1, h2, h3 {
    color: #172B53 !important;
    font-family: 'Inter', sans-serif !important;
}

.hero {
    background:
        radial-gradient(ellipse at 90% 0%, #E5D9FF 0%, transparent 38%),
        linear-gradient(110deg, #F9E9F5, #DFEAFF 65%, #E0E3FF);
    border: 1px solid #D9DFFF;
    border-radius: 19px;
    padding: 30px;
    margin-bottom: 23px;
}

.hero-tag {
    font-size: 11px;
    color: #6A56C4;
    font-weight: 800;
    letter-spacing: 1.5px;
}

.hero h1 {
    color: #17264B !important;
    font-size: clamp(27px, 3vw, 39px);
    line-height: 1.2;
    margin: 12px 0;
    letter-spacing: -1px;
}

.hero p {
    color: #526A93;
    font-size: 14px;
    line-height: 1.8;
    max-width: 720px;
    margin-bottom: 0;
}

.panel {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 3px 14px rgba(35, 63, 112, 0.035);
}

.panel-title {
    font-size: 18px;
    font-weight: 750;
    color: #1A2E54;
    margin-bottom: 6px;
}

.panel-subtitle {
    font-size: 12px;
    color: var(--muted);
    line-height: 1.7;
    margin-bottom: 18px;
}

.section-label {
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.5px;
    color: #7889AA;
    margin: 17px 0 10px;
}

.metric {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 15px;
    min-height: 84px;
}

.metric-label {
    color: #7586A6;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.8px;
}

.metric-value {
    color: #24477D;
    font-size: 21px;
    font-weight: 800;
    margin-top: 7px;
}

.result-box {
    background: linear-gradient(130deg, #F6F8FF, #FCFAFF);
    border: 1px solid #DFE6F7;
    border-left: 4px solid #7963F4;
    border-radius: 11px;
    padding: 20px;
    color: #293C60;
    font-size: 15px;
    line-height: 1.9;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}

.tip {
    background: #F1F4FF;
    border: 1px solid #E0E6FF;
    border-radius: 10px;
    padding: 13px;
    color: #526996;
    font-size: 12px;
    line-height: 1.8;
}

.stButton > button,
.stDownloadButton > button {
    border: 1px solid #D8E3F5;
    border-radius: 9px;
    min-height: 42px;
    font-weight: 650;
    background: white;
    color: #24416F;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: #8C7AF4;
    background: #F3F1FF;
    color: #5C48D4;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(100deg, #7559EF, #4387F3);
    color: white;
    border: none;
    min-height: 48px;
}

.stButton > button[kind="primary"]:hover {
    color: white;
    background: linear-gradient(100deg, #6246DD, #3274E5);
}

.stTextArea textarea {
    background: white !important;
    color: #263B5E !important;
    border: 1px solid #D8E3F5 !important;
    border-radius: 10px !important;
    line-height: 1.75 !important;
}

div[data-baseweb="select"] > div {
    background: white;
    border-color: #D8E3F5;
}

div[data-testid="stRadio"] > div {
    gap: 8px;
}

[data-testid="stSidebar"] .stButton > button {
    background: #192746;
    color: white;
    border-color: #2C3E64;
}

.footer {
    text-align: center;
    color: #8291AB;
    font-size: 11px;
    padding-top: 22px;
}

hr {
    border-color: #E1E8F5;
}

@media (max-width: 700px) {
    .block-container {
        padding: 1rem 0.7rem 2rem;
    }
    .hero {
        padding: 20px;
    }
    .panel {
        padding: 14px;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. SESSION STATE
# =========================================================
if "active_page" not in st.session_state:
    st.session_state.active_page = "Home"

if "prompt" not in st.session_state:
    st.session_state.prompt = ""

if "generated_text" not in st.session_state:
    st.session_state.generated_text = ""

if "generation_history" not in st.session_state:
    st.session_state.generation_history = []

if "selected_example" not in st.session_state:
    st.session_state.selected_example = "Choose an example"


# =========================================================
# 4. LOAD MODEL FILES
# =========================================================
@st.cache_resource
def load_artifacts():
    base_dir = Path(__file__).resolve().parent

    model_path = base_dir / "lstm_model.h5"
    tokenizer_path = base_dir / "token.pickle"
    length_path = base_dir / "max_len.pickle"

    for path in [model_path, tokenizer_path, length_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path.name}"
            )

    model = tf.keras.models.load_model(
        str(model_path),
        compile=False
    )

    # Pickle files must come from a trusted source.
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    with open(length_path, "rb") as f:
        max_len = int(pickle.load(f))

    if max_len <= 0:
        raise ValueError("max_len must be greater than zero.")

    return model, tokenizer, max_len


model = None
tokenizer = None
max_len = None
model_error = None

try:
    model, tokenizer, max_len = load_artifacts()
except Exception as exc:
    model_error = str(exc)


# =========================================================
# 5. TEXT GENERATION
# =========================================================
def predict_next_word(model, tokenizer, text, max_len, temperature):
    sequence = tokenizer.texts_to_sequences([text])[0]

    if not sequence:
        return None

    sequence = sequence[-max_len:]

    padded = pad_sequences(
        [sequence],
        maxlen=max_len,
        padding="pre",
        truncating="pre"
    )

    predictions = np.asarray(
        model.predict(padded, verbose=0)[0],
        dtype=np.float64
    ).reshape(-1)

    if not np.all(np.isfinite(predictions)):
        raise ValueError("Model predictions contain invalid values.")

    # This sampling assumes the model outputs probabilities.
    # If the model outputs raw logits, its sampling needs adjustment.
    scores = np.log(np.maximum(predictions, 1e-12))
    scores /= max(float(temperature), 0.05)
    scores -= np.max(scores)

    probabilities = np.exp(scores)
    probabilities /= probabilities.sum()

    valid_ids = sorted({
        int(idx)
        for idx in tokenizer.word_index.values()
        if 0 < int(idx) < len(probabilities)
    })

    if not valid_ids:
        return None

    valid_probabilities = probabilities[valid_ids]
    valid_probabilities /= valid_probabilities.sum()

    next_id = int(
        np.random.choice(valid_ids, p=valid_probabilities)
    )

    return tokenizer.index_word.get(next_id)


def generate_text(prompt, word_count, temperature):
    text = prompt.strip()

    for _ in range(word_count):
        next_word = predict_next_word(
            model,
            tokenizer,
            text,
            max_len,
            temperature
        )

        if not next_word:
            break

        text += " " + next_word

    return text


# =========================================================
# 6. SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        "<div style='font-size:30px;'>🧠</div>"
        "<div style='font-size:23px;font-weight:800;'>"
        "LSTM Studio</div>"
        "<div style='font-size:12px;color:#A8B9DA;'>"
        "AI Text Generation</div>",
        unsafe_allow_html=True
    )

    st.divider()
    st.markdown("### Generation Settings")

    word_count = st.slider(
        "Words to generate",
        min_value=5,
        max_value=100,
        value=25,
        step=5
    )

    temperature = st.slider(
        "Creativity",
        min_value=0.2,
        max_value=1.5,
        value=0.8,
        step=0.1
    )

    if temperature <= 0.5:
        creativity_text = "More predictable"
    elif temperature <= 1.0:
        creativity_text = "Balanced"
    else:
        creativity_text = "More varied"

    st.caption(
        f"Temperature: {temperature:.1f} · {creativity_text}"
    )

    st.divider()
    st.markdown("### Model Status")

    if model is not None:
        st.success("Model loaded successfully")
    else:
        st.warning("Model files need attention")

    st.caption("Runs locally with your Python environment.")


# =========================================================
# 7. TOP MENU BAR
# =========================================================
st.markdown("""
<div style="
    background:#FFFFFF;
    border:1px solid #DDE6F6;
    border-radius:13px;
    padding:14px 18px;
    margin-bottom:18px;
">
    <span style="font-size:19px;font-weight:800;color:#172B53;">
        🧠 LSTM Studio
    </span>
    <span style="color:#8291AD;font-size:12px;margin-left:12px;">
        AI TEXT GENERATION WORKSPACE
    </span>
</div>
""", unsafe_allow_html=True)

menu_items = [
    "🏠 Home",
    "✨ Text Generator",
    "🧠 Model Info",
    "🕘 History",
    "ℹ️ About"
]

page_names = [
    "Home",
    "Text Generator",
    "Model Info",
    "History",
    "About"
]

current_index = page_names.index(st.session_state.active_page)

selected_menu = st.radio(
    "Main menu",
    menu_items,
    index=current_index,
    horizontal=True,
    label_visibility="collapsed",
    key="main_menu"
)

selected_page = page_names[menu_items.index(selected_menu)]

if selected_page != st.session_state.active_page:
    st.session_state.active_page = selected_page
    st.rerun()

st.divider()


# =========================================================
# 8. REUSABLE HERO
# =========================================================
def show_hero(title, description, tag="YOUR AI WRITING STUDIO"):
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-tag">{html.escape(tag)}</div>
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_model_error():
    if model_error:
        st.error("The LSTM model could not be loaded.")
        st.write(
            "Check that the model, tokenizer, and sequence-length "
            "files exist beside app.py."
        )
        with st.expander("Technical details"):
            st.code(model_error)


# =========================================================
# 9. HOME PAGE
# =========================================================
if st.session_state.active_page == "Home":

    show_hero(
        "Welcome to LSTM Text Generator",
        "Transform your starting ideas into text using your "
        "trained Long Short-Term Memory neural network."
    )

    if model is None:
        show_model_error()
    else:
        c1, c2, c3 = st.columns(3)

        metrics = [
            (c1, "MODEL STATUS", "Ready"),
            (c2, "VOCABULARY", f"{len(tokenizer.word_index):,}"),
            (c3, "INPUT LENGTH", f"{max_len} tokens"),
        ]

        for col, label, value in metrics:
            with col:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
        <div class="panel-title">Welcome to your AI workspace</div>
        <div class="panel-subtitle">
            Generate text, inspect the model and revisit your
            previous generations using the top navigation menu.
        </div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown("""
        <div class="panel">
            <div class="panel-title">✨ Text Generator</div>
            <div class="panel-subtitle">
                Enter a starting sentence and generate a continuation.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Open Text Generator →",
            key="home_generator",
            use_container_width=True
        ):
            st.session_state.active_page = "Text Generator"
            st.rerun()

    with right:
        st.markdown("""
        <div class="panel">
            <div class="panel-title">🧠 Model Information</div>
            <div class="panel-subtitle">
                View the model's vocabulary, input length and output size.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "View Model Information →",
            key="home_model_info",
            use_container_width=True
        ):
            st.session_state.active_page = "Model Info"
            st.rerun()


# =========================================================
# 10. TEXT GENERATOR PAGE
# =========================================================
elif st.session_state.active_page == "Text Generator":

    show_hero(
        "Create something amazing",
        "Enter a prompt, adjust the generation settings and "
        "let your LSTM model continue the text.",
        "TEXT GENERATION"
    )

    show_model_error() if model is None else None

    if model is not None:
        left, right = st.columns([1.6, 1], gap="large")

        with left:
            st.markdown("""
            <div class="panel">
                <div class="panel-title">✍️ Your Starting Prompt</div>
                <div class="panel-subtitle">
                    Choose an example or write your own sentence.
                </div>
            </div>
            """, unsafe_allow_html=True)

            examples = {
                "Choose an example": "",
                "Technology": "The future of artificial intelligence",
                "Story": "Once upon a time in a small village",
                "Science": "Scientists discovered a new way to",
                "Motivation": "Success comes to those who"
            }

            chosen_example = st.selectbox(
                "Example prompts",
                list(examples.keys()),
                key="example_selector"
            )

            if (
                chosen_example != "Choose an example"
                and st.session_state.get("last_prompt_example")
                != chosen_example
            ):
                st.session_state.prompt = examples[chosen_example]

            if chosen_example != "Choose an example":
                st.session_state.last_prompt_example = chosen_example

            prompt = st.text_area(
                "Starting text",
                key="prompt",
                height=150,
                placeholder="Type your starting sentence here..."
            )

            st.caption(f"{len(prompt.split())} words in your prompt")

            if st.button(
                "✨ Generate Text",
                type="primary",
                use_container_width=True
            ):
                if not prompt.strip():
                    st.warning("Please enter a starting prompt.")
                elif not tokenizer.texts_to_sequences([prompt])[0]:
                    st.warning(
                        "The tokenizer does not recognize the words "
                        "in this prompt. Try a different sentence."
                    )
                else:
                    try:
                        with st.spinner("Generating text..."):
                            generated = generate_text(
                                prompt,
                                word_count,
                                temperature
                            )

                        st.session_state.generated_text = generated

                        st.session_state.generation_history.append({
                            "prompt": prompt,
                            "output": generated
                        })

                        st.rerun()

                    except Exception as exc:
                        st.error("Generation failed.")
                        st.exception(exc)

        with right:
            st.markdown("""
            <div class="panel">
                <div class="panel-title">🎛️ Generation Settings</div>
                <div class="panel-subtitle">
                    These values come from the sidebar.
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(
                f"<div class='metric'>"
                f"<div class='metric-label'>OUTPUT LENGTH</div>"
                f"<div class='metric-value'>{word_count} words</div>"
                f"</div>",
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(
                f"<div class='metric'>"
                f"<div class='metric-label'>CREATIVITY</div>"
                f"<div class='metric-value'>{temperature:.1f} / 1.5</div>"
                f"</div>",
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("""
            <div class="tip">
                <b>Helpful tips</b><br><br>
                • Start with a short, clear sentence.<br>
                • Experiment with different prompts.<br>
                • Compare results at different temperatures.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Generated Output")

        if st.session_state.generated_text:
            st.markdown(
                "<div class='result-box'>"
                + html.escape(st.session_state.generated_text)
                + "</div>",
                unsafe_allow_html=True
            )

            edited_output = st.text_area(
                "Edit your generated text",
                value=st.session_state.generated_text,
                height=150,
                key="edited_output"
            )

            st.caption(
                f"{len(edited_output.split())} words · "
                f"{len(edited_output)} characters"
            )

            d1, d2 = st.columns(2)

            with d1:
                st.download_button(
                    "⬇ Download Text",
                    data=edited_output,
                    file_name="generated_text.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            with d2:
                if st.button(
                    "Clear Output",
                    use_container_width=True
                ):
                    st.session_state.generated_text = ""
                    st.session_state.pop("edited_output", None)
                    st.rerun()
        else:
            st.info(
                "Your generated text will appear here after you "
                "click Generate Text."
            )


# =========================================================
# 11. MODEL INFORMATION PAGE
# =========================================================
elif st.session_state.active_page == "Model Info":

    show_hero(
        "Model Information",
        "Explore the configuration and vocabulary of your trained LSTM."
    )

    if model is None:
        show_model_error()
    else:
        st.markdown("### Model Overview")

        info = [
            ("Architecture", "LSTM"),
            ("Vocabulary size", f"{len(tokenizer.word_index):,} words"),
            ("Input sequence length", f"{max_len} tokens"),
            ("Output dimension", str(model.output_shape[-1])),
            ("Number of layers", str(len(model.layers))),
            ("Trainable parameters", f"{model.count_params():,}"),
        ]

        for i in range(0, len(info), 2):
            cols = st.columns(2)

            for j, col in enumerate(cols):
                idx = i + j

                if idx < len(info):
                    label, value = info[idx]

                    with col:
                        st.markdown(
                            f"<div class='metric'>"
                            f"<div class='metric-label'>"
                            f"{html.escape(label.upper())}</div>"
                            f"<div class='metric-value'>"
                            f"{html.escape(value)}</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

            st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("View model layer summary"):
            try:
                rows = []

                for layer in model.layers:
                    rows.append({
                        "Layer": layer.name,
                        "Type": layer.__class__.__name__,
                        "Output": str(layer.output.shape)
                    })

                st.dataframe(rows, use_container_width=True)

            except Exception as exc:
                st.warning(f"Could not display layer details: {exc}")

        st.caption(
            "Embedding size is not assumed here; inspect the Embedding "
            "layer to determine its actual dimensions."
        )


# =========================================================
# 12. HISTORY PAGE
# =========================================================
elif st.session_state.active_page == "History":

    show_hero(
        "Generation History",
        "Review previous text generations from your current session.",
        "YOUR WORKSPACE"
    )

    history = st.session_state.generation_history

    if not history:
        st.info(
            "No generations yet. Visit Text Generator to create your "
            "first result."
        )
    else:
        st.write(f"**{len(history)} generations in this session**")

        if st.button("Clear All History"):
            st.session_state.generation_history = []
            st.rerun()

        for index, item in enumerate(reversed(history), start=1):
            with st.expander(f"Generation {index}", expanded=(index == 1)):
                st.markdown("**Starting prompt**")
                st.write(item["prompt"])

                st.markdown("**Generated output**")
                st.write(item["output"])

                st.download_button(
                    "Download this generation",
                    data=item["output"],
                    file_name=f"generation_{index}.txt",
                    mime="text/plain",
                    key=f"download_history_{index}"
                )


# =========================================================
# 13. ABOUT PAGE
# =========================================================
elif st.session_state.active_page == "About":

    show_hero(
        "About LSTM Studio",
        "A simple, interactive interface for experimenting with "
        "neural-network-based next-word prediction.",
        "ABOUT THIS PROJECT"
    )

    st.markdown("""
    <div class="panel">
        <div class="panel-title">What is an LSTM?</div>
        <div class="panel-subtitle">
            Long Short-Term Memory (LSTM) is a type of recurrent neural
            network that learns patterns in sequences. In a text
            generator, it can use previous words to predict a likely
            next word.
        </div>
    </div>

    <div class="panel">
        <div class="panel-title">How this application works</div>
        <div class="panel-subtitle">
            1. You enter a starting prompt.<br>
            2. The tokenizer converts words to integer IDs.<br>
            3. The LSTM predicts the next word.<br>
            4. The predicted word is appended to the prompt.<br>
            5. The process repeats until the requested length is reached.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div class='tip'>"
        "Note: The quality of generated text depends on the training "
        "data, model architecture, vocabulary and training process."
        "</div>",
        unsafe_allow_html=True
    )


# =========================================================
# 14. FOOTER
# =========================================================
st.markdown("---")

st.markdown("""
<div class="footer">
    LSTM STUDIO · BUILT WITH PYTHON AND STREAMLIT
</div>
""", unsafe_allow_html=True)
