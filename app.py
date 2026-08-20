import streamlit as st
from agents import (
    build_readeragent,
    build_searchagent,
    writer_chain,
    critic_chain,
)

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔎",
    layout="wide",
)

# -----------------------------
# CUSTOM CSS
# -----------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        color: #888;
        margin-bottom: 30px;
    }

    .step-card {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-bottom: 10px;
    }

    .report-box {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #ddd;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# HEADER
# -----------------------------

st.markdown(
    '<div class="main-title">🔎 AI Research Agent</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Multi-agent research system that searches, scrapes, writes and critiques."
    "</div>",
    unsafe_allow_html=True,
)

# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:

    st.header("⚙️ Pipeline")

    st.markdown(
        """
        **1️⃣ Search Agent**

        Finds recent and reliable information.

        **2️⃣ Reader Agent**

        Selects and scrapes the most relevant source.

        **3️⃣ Writer Chain**

        Generates the research report.

        **4️⃣ Critic Chain**

        Reviews the generated report.
        """
    )

    st.divider()

    st.info(
        "The agents work sequentially. "
        "Each stage passes information to the next stage."
    )


# -----------------------------
# PIPELINE FUNCTION
# -----------------------------

def run_pipeline(topic):

    state = {}

    # Progress UI
    progress = st.progress(0)

    status = st.empty()

    # =================================
    # STEP 1 — SEARCH AGENT
    # =================================

    status.info("🔎 Step 1/4 — Search Agent is researching...")

    searchagent = build_searchagent()

    searchresult = searchagent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"Find recent reliable and detailed information about the {topic}",
                )
            ]
        }
    )

    state["searchresult"] = searchresult["messages"][-1].content

    progress.progress(25)

    # =================================
    # STEP 2 — READER AGENT
    # =================================

    status.info(
        "📖 Step 2/4 — Reader Agent is analyzing the best source..."
    )

    reader_agent = build_readeragent()

    reader_result = reader_agent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"""
Based on the following search results about '{topic}',
pick the most relevant URL and scrape it for deeper content.

Search Results:

{state['searchresult'][:800]}
""",
                )
            ]
        }
    )

    state["scraped_content"] = (
        reader_result["messages"][-1].content
    )

    progress.progress(50)

    # =================================
    # STEP 3 — WRITER
    # =================================

    status.info("✍️ Step 3/4 — Writer is drafting the report...")

    research_combined = (
        f"SEARCH RESULTS:\n{state['searchresult']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke(
        {
            "topic": topic,
            "research": research_combined,
        }
    )

    progress.progress(75)

    # =================================
    # STEP 4 — CRITIC
    # =================================

    status.info("🧐 Step 4/4 — Critic is reviewing the report...")

    state["feedback"] = critic_chain.invoke(
        {
            "report": state["report"],
        }
    )

    progress.progress(100)

    status.success("✅ Research completed!")

    return state


# -----------------------------
# USER INPUT
# -----------------------------

st.subheader("What do you want to research?")

topic = st.text_input(
    "Research Topic",
    placeholder="Example: Latest developments in Agentic AI",
)

run_button = st.button(
    "🚀 Start Research",
    type="primary",
    use_container_width=True,
)


# -----------------------------
# RUN PIPELINE
# -----------------------------

if run_button:

    if not topic.strip():

        st.warning("⚠️ Please enter a research topic.")

    else:

        try:

            with st.spinner("AI agents are working..."):

                result = run_pipeline(topic.strip())

            # -----------------------------
            # METRICS
            # -----------------------------

            st.divider()

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Search Agent",
                "✓ Done",
            )

            col2.metric(
                "Reader Agent",
                "✓ Done",
            )

            col3.metric(
                "Writer",
                "✓ Done",
            )

            col4.metric(
                "Critic",
                "✓ Done",
            )

            st.divider()

            # -----------------------------
            # TABS
            # -----------------------------

            tab1, tab2, tab3 = st.tabs(
                [
                    "📄 Final Report",
                    "🧐 Critic Feedback",
                    "🔎 Research Data",
                ]
            )

            # -----------------------------
            # REPORT
            # -----------------------------

            with tab1:

                st.subheader("📄 Final Research Report")

                st.markdown(result["report"])

                st.download_button(
                    label="⬇️ Download Report",
                    data=result["report"],
                    file_name=(
                        f"{topic.replace(' ', '_')}_report.txt"
                    ),
                    mime="text/plain",
                )

            # -----------------------------
            # CRITIC
            # -----------------------------

            with tab2:

                st.subheader("🧐 Critic Feedback")

                st.markdown(result["feedback"])

            # -----------------------------
            # RAW RESEARCH
            # -----------------------------

            with tab3:

                st.subheader("🔎 Search Results")

                with st.expander(
                    "View Search Agent Results"
                ):
                    st.write(
                        result["searchresult"]
                    )

                st.subheader(
                    "📖 Scraped Content"
                )

                with st.expander(
                    "View Reader Agent Output"
                ):
                    st.write(
                        result["scraped_content"]
                    )

        except Exception as e:

            st.error(
                "❌ Something went wrong while running the pipeline."
            )

            st.exception(e)