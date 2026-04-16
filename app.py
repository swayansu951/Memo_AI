"""
app.py — Voice-Controlled Local AI Agent
Streamlit front-end that wires together STT → Intent → Tools → Display.
"""

import streamlit as st
import tempfile, os, time
from pathlib import Path

from agent.stt import STTEngine
from agent.intent_classifier import SUPERVISOR, AGENTS, EVALUATOR
from agent.memory import SessionMemory

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Voice AI Agent",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS  (dark glassmorphism theme)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ---------- global reset ---------- */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0d0f1a; color: #e2e8f0; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12152b 0%, #0d0f1a 100%);
    border-right: 1px solid #1e2340;
}

/* ---------- cards ---------- */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(8px);
}
.card-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: .5rem;
}
.card-body { font-size: 0.95rem; line-height: 1.6; color: #cbd5e1; }

/* ---------- intent badges ---------- */
.badge {
    display: inline-block;
    padding: .25rem .7rem;
    border-radius: 999px;
    font-size: .75rem;
    font-weight: 600;
    letter-spacing: .05em;
}
.badge-write_code   { background:#1e3a5f; color:#60a5fa; }
.badge-create_file  { background:#163832; color:#34d399; }
.badge-summarize    { background:#3b2065; color:#c084fc; }
.badge-chat         { background:#3b280a; color:#fb923c; }
.badge-error        { background:#3b0a0a; color:#f87171; }

/* ---------- code block ---------- */
.code-block {
    background: #0a0c16;
    border: 1px solid #1e2340;
    border-radius: 10px;
    padding: 1rem;
    font-family: 'Fira Code', monospace;
    font-size: .82rem;
    color: #a5f3fc;
    overflow-x: auto;
    white-space: pre-wrap;
}

/* ---------- log row ---------- */
.log-row {
    display: flex;
    gap: .8rem;
    align-items: flex-start;
    padding: .6rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: .85rem;
}
.log-time  { color: #475569; min-width: 55px; }
.log-badge { min-width: 90px; }

/* ---------- buttons ---------- */
.stButton > button {
    background: linear-gradient(135deg,#6366f1,#8b5cf6);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    transition: opacity .2s;
}
.stButton > button:hover { opacity: .85; }

/* ---------- upload area ---------- */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03);
    border: 1px dashed #334155;
    border-radius: 12px;
    padding: .5rem;
}

/* ---------- spinner ---------- */
.stSpinner > div { border-top-color: #6366f1 !important; }

/* ---------- section header ---------- */
.section-hdr {
    font-size: 1.35rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: .5rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialisation
# ─────────────────────────────────────────────────────────────────────────────
if "memory" not in st.session_state:
    st.session_state.memory = SessionMemory()
if "stt_engine" not in st.session_state:
    st.session_state.stt_engine = None          # lazy-loaded on first use
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None     # last run's full output

memory: SessionMemory = st.session_state.memory


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load STT model (cached per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_stt_engine(model_path: str) -> STTEngine:
    return STTEngine(model_size=model_path)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: render intent badge
# ─────────────────────────────────────────────────────────────────────────────
def intent_badge(intent: str) -> str:
    labels = {
        "write_code":  "⌨️ Write Code",
        "create_file": "📄 Create File",
        "summarize":   "📝 Summarize",
        "chat":        "💬 Chat",
    }
    label = labels.get(intent, intent)
    cls   = f"badge-{intent}" if intent in labels else "badge-error"
    return f'<span class="badge {cls}">{label}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Helper: run full pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(audio_bytes: bytes, sup_model: str, agent_model: str, human_confirm: bool):
    result_bundle = {}

    # 1. Save audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # 2. STT
        with st.status("🎙️ Transcribing audio…", expanded=False) as s:
            stt: STTEngine = load_stt_engine(st.session_state.whisper_model_path)
            transcription  = stt.transcribe(tmp_path)
            s.update(label=f"✅ Transcribed — {len(transcription)} chars", state="complete")

        result_bundle["transcription"] = transcription
        if not transcription:
            st.error("⚠️ Could not transcribe audio. Please try again.")
            return

        # 3. Intent Classification
        with st.status("🧠 Classifying intent…", expanded=False) as s:
            supervisor   = SUPERVISOR(transcription)
            intent_dict  = supervisor.classify(model=sup_model)
            valid, msg   = EVALUATOR.validate_intent(intent_dict)
            if not valid:
                intent_dict = {"intent": "chat", "params": {"message": transcription}}
            s.update(label=f"✅ Intent: {intent_dict['intent']}", state="complete")

        intent = intent_dict["intent"]
        params = intent_dict.get("params", {})
        result_bundle["intent"] = intent
        result_bundle["params"] = params

        # 4. Human-in-the-Loop (optional)
        if human_confirm and intent in ("create_file", "write_code"):
            st.warning(
                f"⚠️ **Confirm action:** The agent wants to **{intent}**\n\n"
                f"Params: `{params}`"
            )
            col1, col2 = st.columns(2)
            confirmed = col1.button("✅ Allow", key="confirm_yes")
            cancelled = col2.button("❌ Cancel", key="confirm_no")
            if cancelled:
                st.info("Action cancelled by user.")
                return
            if not confirmed:
                st.stop()  # wait for user to click

        # 5. Tool Execution
        agents = AGENTS(model=agent_model)
        if intent == "chat":
            with st.status("💬 Generating response…", expanded=False) as s:
                history = memory.get_history()
                stream  = agents.run(intent, params, history=history)
                full_reply = ""
                reply_box  = st.empty()
                for chunk in stream:
                    full_reply += chunk
                    reply_box.markdown(
                        f'<div class="card"><div class="card-title">Assistant</div>'
                        f'<div class="card-body">{full_reply}</div></div>',
                        unsafe_allow_html=True,
                    )
                result_bundle["result"] = {"status": "success", "action": "chat", "reply": full_reply}
                memory.add_user_message(transcription)
                memory.add_assistant_message(full_reply)
                s.update(label="✅ Response ready", state="complete")
        else:
            with st.status(f"⚙️ Executing {intent}…", expanded=False) as s:
                tool_result = agents.run(intent, params)
                result_bundle["result"] = tool_result
                s.update(
                    label=f"{'✅' if tool_result['status']=='success' else '❌'} {tool_result['message']}",
                    state="complete" if tool_result["status"] == "success" else "error",
                )
            # Also add to chat memory context
            memory.add_user_message(transcription)
            memory.add_assistant_message(tool_result.get("message", ""))

        # 6. Log the action
        memory.log_action(
            transcription=transcription,
            intent=intent,
            params=params,
            result=result_bundle.get("result", {}),
        )

        st.session_state.pipeline_result = result_bundle

    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    st.markdown("**Whisper Model Path**")
    whisper_path = st.text_input(
        label="whisper_path",
        value="base",
        label_visibility="collapsed",
        key="whisper_model_path",
    )

    st.markdown("**Supervisor LLM**")
    sup_model = st.selectbox(
        label="sup_model",
        options=[
            "llama3.1:8b-instruct-q5_K_S",
            "llama3:latest",
            "mistral:latest",
            "phi3:mini",
        ],
        label_visibility="collapsed",
        key="sup_model",
    )

    st.markdown("**Agent / Tool LLM**")
    agent_model = st.selectbox(
        label="agent_model",
        options=[
            "llama3-abliterated:latest",
            "llama3:latest",
            "mistral:latest",
            "codellama:latest",
        ],
        label_visibility="collapsed",
        key="agent_model",
    )

    st.divider()
    human_confirm = st.toggle(
        "🔒 Human-in-the-Loop",
        value=True,
        help="Ask for confirmation before creating/writing files.",
    )

    st.divider()
    if st.button("🗑️ Clear Session", use_container_width=True):
        memory.clear_all()
        st.session_state.pipeline_result = None
        st.success("Session cleared.")
        time.sleep(0.5)
        st.rerun()

    st.divider()
    st.markdown(
        '<div style="color:#475569;font-size:.78rem;text-align:center;">'
        'Mem0 AI Intern Assignment<br>Voice-Controlled Local Agent'
        '</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main Layout
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-hdr">🎙️ Voice-Controlled <span style="color:#6366f1;">AI Agent</span></div>',
    unsafe_allow_html=True,
)

tab_input, tab_result, tab_log = st.tabs(["🎙️ Audio Input", "📊 Results", "📋 Action Log"])

# ── Tab 1: Audio Input ─────────────────────────────────────────────────────
with tab_input:
    col_mic, col_file = st.columns([1, 1], gap="large")

    with col_mic:
        st.markdown("### 🎤 Record from Microphone")
        try:
            from audio_recorder_streamlit import audio_recorder
            audio_bytes_mic = audio_recorder(
                text="Click to record",
                recording_color="#6366f1",
                neutral_color="#475569",
                icon_name="microphone",
                icon_size="2x",
            )
        except ImportError:
            st.info(
                "Install `audio-recorder-streamlit` to enable microphone recording:\n"
                "```\npip install audio-recorder-streamlit\n```"
            )
            audio_bytes_mic = None

    with col_file:
        st.markdown("### 📁 Upload Audio File")
        uploaded = st.file_uploader(
            "Drop a .wav or .mp3 file",
            type=["wav", "mp3"],
            label_visibility="collapsed",
        )
        audio_bytes_file = uploaded.read() if uploaded else None

    # Decide which audio source to use
    audio_source = audio_bytes_mic or audio_bytes_file

    if audio_source:
        st.audio(audio_source, format="audio/wav")

    st.divider()
    btn_run = st.button(
        "🚀 Run Agent Pipeline",
        use_container_width=True,
        disabled=(audio_source is None),
    )

    if btn_run and audio_source:
        with tab_result:
            run_pipeline(
                audio_bytes=audio_source,
                sup_model=st.session_state.sup_model,
                agent_model=st.session_state.agent_model,
                human_confirm=human_confirm,
            )
        # Switch to results tab
        st.rerun()

# ── Tab 2: Results ─────────────────────────────────────────────────────────
with tab_result:
    pr = st.session_state.pipeline_result

    if pr is None:
        st.markdown(
            '<div class="card" style="text-align:center;padding:2rem;">'
            '<div style="font-size:2.5rem">🎙️</div>'
            '<div class="card-body" style="color:#475569;margin-top:.5rem;">'
            'Upload or record audio and click <b>Run Agent Pipeline</b> to see results here.'
            '</div></div>',
            unsafe_allow_html=True,
        )
    else:
        # Transcription
        st.markdown(
            f'<div class="card">'
            f'<div class="card-title">📝 Transcription</div>'
            f'<div class="card-body">{pr.get("transcription","—")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Intent
        intent = pr.get("intent", "unknown")
        params = pr.get("params", {})
        st.markdown(
            f'<div class="card">'
            f'<div class="card-title">🧠 Detected Intent</div>'
            f'<div class="card-body">'
            f'{intent_badge(intent)}'
            f'<br><br><span style="color:#64748b;font-size:.82rem;">Parameters</span><br>'
            f'<code style="color:#94a3b8;">{params}</code>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # Result
        result = pr.get("result", {})
        status_icon = "✅" if result.get("status") == "success" else "❌"
        action = result.get("action", intent)

        st.markdown(
            f'<div class="card">'
            f'<div class="card-title">⚙️ Action Taken &nbsp; {status_icon} {action}</div>'
            f'<div class="card-body">{result.get("message","")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Extra output depending on action type
        if action == "write_code" and result.get("code"):
            st.markdown(
                f'<div class="card-title" style="margin-top:.5rem;">Generated Code</div>'
                f'<div class="code-block">{result["code"]}</div>',
                unsafe_allow_html=True,
            )

        elif action == "summarize" and result.get("summary"):
            st.markdown(
                f'<div class="card">'
                f'<div class="card-title">📝 Summary</div>'
                f'<div class="card-body">{result["summary"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        elif action == "chat" and result.get("reply"):
            pass  # Already rendered inline during streaming

# ── Tab 3: Action Log ──────────────────────────────────────────────────────
with tab_log:
    log = memory.get_log()
    if not log:
        st.markdown(
            '<div class="card" style="text-align:center;padding:2rem;">'
            '<div style="font-size:2rem">📋</div>'
            '<div class="card-body" style="color:#475569;margin-top:.5rem;">'
            'No actions yet. Run the pipeline to see your history here.'
            '</div></div>',
            unsafe_allow_html=True,
        )
    else:
        for entry in reversed(log):
            res  = entry.get("result", {})
            stat = "✅" if res.get("status") == "success" else "❌"
            st.markdown(
                f'<div class="log-row">'
                f'  <span class="log-time">{entry["timestamp"]}</span>'
                f'  <span class="log-badge">{intent_badge(entry["intent"])}</span>'
                f'  <span style="color:#e2e8f0;">{stat} {entry["transcription"][:90]}…</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # Expandable detail view
        with st.expander("🔍 Full Log Details"):
            for i, entry in enumerate(reversed(log)):
                st.json(entry)
