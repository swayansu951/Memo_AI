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

/* ---------- custom tabs (radio override) ---------- */
div[data-testid="stRadio"] > div {
    flex-direction: row;
    background: rgba(255,255,255,0.03);
    padding: 4px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 1.5rem;
}
div[data-testid="stRadio"] label {
    background: transparent;
    border: none;
    padding: 6px 16px;
    border-radius: 8px;
    color: #94a3b8;
    cursor: pointer;
    transition: all 0.2s;
}
div[data-testid="stRadio"] label[data-checked="true"] {
    background: #6366f1;
    color: white;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}
div[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] p {
    font-size: 0.88rem;
}
/* hide the radio circle */
div[data-testid="stRadio"] label span[data-testid="stWidgetLabel"] {
    display: none;
}
div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
    display: none;
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
if "pending_confirmation" not in st.session_state:
    st.session_state.pending_confirmation = False
if "metrics" not in st.session_state:
    st.session_state.metrics = {"stt": 0.0, "intent": 0.0, "tool": 0.0}
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "🎙️ Audio Input"

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
def execute_tool_action(bundle, agent_model, memory):
    """Executes the tool part of the pipeline (Step 5) for multiple intents."""
    intents = bundle["intents"]
    transcription = bundle["transcription"]
    
    agents = AGENTS(model=agent_model)
    bundle["results"] = []
    
    total_tool_time = 0
    
    for i, item in enumerate(intents):
        intent = item["intent"]
        params = item["params"]
        
        # If it's the second+ tool and it needs content from the first (e.g. summary)
        # simplistic placeholder replacement
        if i > 0 and "content" in params and "SUMMARY_PLACEHOLDER" in params["content"]:
            prev_res = bundle["results"][i-1]
            if prev_res.get("action") == "summarize" and "summary" in prev_res:
                params["content"] = prev_res["summary"]

        if intent == "chat":
            # Chat is usually handled streaming in run_pipeline, but if part of compound:
            history = memory.get_history()
            # Non-streaming chat for compound consistency
            response = "".join(agents.run(intent, params, history=history))
            res = {"status": "success", "action": "chat", "message": response, "reply": response}
            bundle["results"].append(res)
        else:
            with st.status(f"⚙️ Task {i+1}: Executing {intent}…", expanded=True) as s:
                start_tool = time.time()
                tool_result = agents.run(intent, params)
                total_tool_time += (time.time() - start_tool)
                bundle["results"].append(tool_result)
                s.update(
                    label=f"{'✅' if tool_result['status']=='success' else '❌'} {tool_result['message']}",
                    state="complete" if tool_result["status"] == "success" else "error",
                )
        
        # Log each tool action
        memory.log_action(
            transcription=transcription,
            intent=intent,
            params=params,
            result=bundle["results"][-1],
        )

    st.session_state.metrics["tool"] = total_tool_time
    
    # Update memory with full transcription at the end
    memory.add_user_message(transcription)
    final_responses = [r.get("message", r.get("reply", "")) for r in bundle["results"]]
    memory.add_assistant_message(" | ".join(final_responses))
        
    st.session_state.pipeline_result = bundle
    st.session_state.pending_confirmation = False


def run_pipeline(audio_bytes: bytes, sup_model: str, agent_model: str, human_confirm: bool):
    result_bundle = {}

    # 1. Save audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # 2. STT
        start_stt = time.time()
        with st.status("🎙️ Transcribing audio...", expanded=True) as s:
            stt: STTEngine = load_stt_engine(st.session_state.whisper_model_path)
            segments, info = stt.transcribe(tmp_path)
            
            transcription = ""
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            duration = info.duration
            processed_time = 0
            
            for segment in segments:
                transcription += segment.text + " "
                processed_time = segment.end
                if duration > 0:
                    pct = min(processed_time / duration, 1.0)
                    progress_bar.progress(pct)
                status_text.write(f"Transcribed: {processed_time:.1f}s / {duration:.1f}s")
            
            transcription = transcription.strip()
            st.session_state.metrics["stt"] = time.time() - start_stt
            s.update(label=f"✅ Transcribed — {len(transcription)} chars", state="complete")

        result_bundle["transcription"] = transcription
        if not transcription:
            st.error("⚠️ Could not transcribe audio. Please try again.")
            return

        # 3. Intent Classification
        start_intent = time.time()
        with st.status("🧠 Classifying intent…", expanded=False) as s:
            supervisor   = SUPERVISOR(transcription)
            intent_dict  = supervisor.classify(model=sup_model)
            valid, msg   = EVALUATOR.validate_intent(intent_dict)
            if not valid:
                intent_dict = {"intents": [{"intent": "chat", "params": {"message": transcription}}]}
            st.session_state.metrics["intent"] = time.time() - start_intent
            s.update(label=f"✅ Intent: {len(intent_dict['intents'])} tasks detected", state="complete")

        intents = intent_dict["intents"]
        result_bundle["intents"] = intents

        # 4. Check for Confirmation
        # If ANY intent needs confirmation
        needs_confirm = any(item["intent"] in ("create_file", "write_code") for item in intents)
        if human_confirm and needs_confirm:
            st.session_state.pending_confirmation = True
            st.session_state.pipeline_result = result_bundle
            return

        # 5. Tool Execution
        # If it's a single chat intent, we stream it
        if len(intents) == 1 and intents[0]["intent"] == "chat":
            intent = intents[0]["intent"]
            params = intents[0]["params"]
            start_tool = time.time()
            with st.status("💬 Generating response…", expanded=True) as s:
                agents = AGENTS(model=agent_model)
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
                result_bundle["results"] = [{"status": "success", "action": "chat", "reply": full_reply}]
                memory.add_user_message(transcription)
                memory.add_assistant_message(full_reply)
                st.session_state.metrics["tool"] = time.time() - start_tool
                s.update(label="✅ Response ready", state="complete")
            
            memory.log_action(
                transcription=transcription,
                intent=intent,
                params=params,
                result=result_bundle["results"][0],
            )
            st.session_state.pipeline_result = result_bundle
        else:
            execute_tool_action(result_bundle, agent_model, memory)

    finally:
        if os.path.exists(tmp_path):
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
        value="./whisper_model",  # Default to local folder
        label_visibility="collapsed",
        key="whisper_model_path",
        help="Use 'base', 'small' or a path to a local CTranslate2 model directory."
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
            "llama3.2:3b",
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
        key="human_in_the_loop_toggle",
        help="Ask for confirmation before creating/writing files.",
    )

    st.divider()
    if st.button("🗑️ Clear Session", use_container_width=True):
        memory.clear_all()
        st.session_state.pipeline_result = None
        st.session_state.metrics = {"stt": 0.0, "intent": 0.0, "tool": 0.0}
        st.success("Session cleared.")
        time.sleep(0.5)
        st.rerun()

    st.divider()
    st.markdown("**📊 Model Benchmark**")
    m = st.session_state.metrics
    st.markdown(
        f'<div style="font-size: .85rem; color: #94a3b8; line-height: 1.6;">'
        f'🎙️ STT: <span style="color:#60a5fa; float:right;">{m["stt"]:.2f}s</span><br>'
        f'🧠 Intent ({st.session_state.sup_model}): <span style="color:#c084fc; float:right;">{m["intent"]:.2f}s</span><br>'
        f'⚙️ Tool ({st.session_state.agent_model}): <span style="color:#34d399; float:right;">{m["tool"]:.2f}s</span><br>'
        f'<hr style="margin: .5rem 0; border-color: rgba(255,255,255,0.05);">'
        f'<b>Total: <span style="color:#f1f5f9; float:right;">{sum(m.values()):.2f}s</span></b>'
        f'</div>',
        unsafe_allow_html=True,
    )

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

# ── Global Notification for Pending Action ──────────────────────────────────
if st.session_state.pending_confirmation:
    st.warning(
        "🔒 **Action Required:** The agent is waiting for your approval in the **Results** tab.",
        icon="⚠️"
    )

# ── Dynamic Tab Navigation ──────────────────────────────────────────────────
tabs = ["🎙️ Audio Input", "📊 Results", "📋 Action Log"]

# To allow programmatic switching, we manage the state externally without a forced key.
tab_select = st.radio(
    label="Navigation",
    options=tabs,
    index=tabs.index(st.session_state.active_tab),
    horizontal=True,
    label_visibility="collapsed",
)

# Sync radio selection with our manual state indicator
if tab_select != st.session_state.active_tab:
    st.session_state.active_tab = tab_select
    st.rerun()

active_tab = st.session_state.active_tab

# ── Tab Content ─────────────────────────────────────────────────────────────
if active_tab == "🎙️ Audio Input":
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
        run_pipeline(
            audio_bytes=audio_source,
            sup_model=st.session_state.sup_model,
            agent_model=st.session_state.agent_model,
            human_confirm=human_confirm,
        )
        # Shift tab view AFTER background task processes finish
        st.session_state.active_tab = "📊 Results"
        st.rerun()

elif active_tab == "📊 Results":
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

        # Intents
        intents = pr.get("intents", [])
        results = pr.get("results", [])

        st.markdown('<div class="section-hdr" style="font-size:1rem;margin-top:1.5rem;">📋 Tasks & Results</div>', unsafe_allow_html=True)

        for i, item in enumerate(intents):
            intent = item.get("intent", "unknown")
            params = item.get("params", {})
            result = results[i] if i < len(results) else None

            # Task Card
            with st.container():
                c_task, c_res = st.columns([1, 1], gap="medium")
                
                with c_task:
                    st.markdown(
                        f'<div class="card">'
                        f'<div class="card-title">Task {i+1}: {intent_badge(intent)}</div>'
                        f'<div class="card-body">'
                        f'<span style="color:#64748b;font-size:.82rem;">Parameters</span><br>'
                        f'<code style="color:#94a3b8;word-break:break-all;">{params}</code>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

                with c_res:
                    if result:
                        status_icon = "✅" if result.get("status") == "success" else "❌"
                        st.markdown(
                            f'<div class="card">'
                            f'<div class="card-title">{status_icon} Result</div>'
                            f'<div class="card-body">{result.get("message", result.get("reply", ""))}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        # Specific outputs
                        if result.get("action") == "write_code" and result.get("code"):
                            st.markdown(f'<div class="code-block">{result["code"]}</div>', unsafe_allow_html=True)
                        elif result.get("action") == "summarize" and result.get("summary"):
                            st.info(f"**Summary:** {result['summary']}")
                    else:
                        st.info("Pending execution...")

        # Handle Confirmation UI if pending
        if st.session_state.pending_confirmation:
            st.warning(
                f"⚠️ **Action required:** The agent is waiting for your permission to execute these {len(intents)} tasks."
            )
            c1, c2 = st.columns(2)
            if c1.button("✅ Allow and Execute All", key="final_confirm", use_container_width=True):
                execute_tool_action(pr, st.session_state.agent_model, memory)
                st.rerun()
            if c2.button("❌ Cancel Actions", key="final_cancel", use_container_width=True):
                st.session_state.pending_confirmation = False
                st.session_state.pipeline_result = None
                st.rerun()

elif active_tab == "📋 Action Log":
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
