import streamlit as st

from policy_rag.rag import answer_question


st.set_page_config(
    page_title="Warwick Policy Assistant",
    page_icon="📘",
    layout="centered",
)


HELPFUL_REFUSAL_MESSAGE = (
    "I could not find sufficient evidence in the six policy documents "
    "available to this prototype to answer this question. I can answer "
    "English questions about AI information compliance, user account "
    "management, systems administration, data protection, information "
    "access control, and acceptable use. Please rephrase the question "
    "within this scope or consult the official University of Warwick "
    "policy pages."
)


LEGACY_REFUSAL_MESSAGES = {
    "No directly supported answer was found in the available policies.",
    "I cannot answer this question from the provided policies.",
    "The answer cannot be determined from the available content.",
}


def render_sources(sources: list[dict]) -> None:
    """Display retrieved policy evidence without exposing local file paths."""

    if not sources:
        return

    with st.expander("Verified policy sources"):
        for index, source in enumerate(sources):
            policy = source.get("policy", "Unknown policy")
            section = source.get("section", "Unknown section")
            title = source.get("document_title", "")
            passage = source.get("text", "")
            score = source.get("score")

            st.markdown(f"**{policy}, Section {section}**")

            caption_parts = []

            if title:
                caption_parts.append(str(title))

            if score is not None:
                try:
                    caption_parts.append(
                        f"retrieval score: {float(score):.4f}"
                    )
                except (TypeError, ValueError):
                    pass

            if caption_parts:
                st.caption(" · ".join(caption_parts))

            if passage:
                st.markdown("**Retrieved policy passage**")
                st.write(passage)

            if index < len(sources) - 1:
                st.divider()


def prepare_assistant_message(result: dict) -> dict:
    """Convert the RAG result into a message displayed by Streamlit."""

    if not isinstance(result, dict):
        return {
            "role": "assistant",
            "content": HELPFUL_REFUSAL_MESSAGE,
            "status": "unsupported",
            "sources": [],
        }

    answer = str(result.get("answer", "")).strip()
    status = str(result.get("status", "")).strip().lower()
    sources = result.get("sources", [])

    if not isinstance(sources, list):
        sources = []

    if status == "unsupported" or answer in LEGACY_REFUSAL_MESSAGES:
        return {
            "role": "assistant",
            "content": HELPFUL_REFUSAL_MESSAGE,
            "status": "unsupported",
            "sources": [],
        }

    if not answer:
        return {
            "role": "assistant",
            "content": HELPFUL_REFUSAL_MESSAGE,
            "status": "unsupported",
            "sources": [],
        }

    return {
        "role": "assistant",
        "content": answer,
        "status": status or "supported",
        "sources": sources,
    }


def render_assistant_message(message: dict) -> None:
    """Display a saved assistant response and its policy evidence."""

    status = message.get("status", "unsupported")

    if status == "unsupported":
        st.info(
            message.get(
                "content",
                HELPFUL_REFUSAL_MESSAGE,
            )
        )

    elif status == "error":
        st.error(message.get("content", ""))

    else:
        st.markdown(message.get("content", ""))

        # Display policy evidence only for supported answers.
        render_sources(message.get("sources", []))


with st.sidebar:
    st.subheader("System information")

    st.markdown("**Retrieval:** MiniLM semantic search")
    st.markdown("**Retrieval setting:** Top-K = 1")
    st.markdown("**Local model:** Llama3.2:3B")
    st.markdown("**Language:** English questions only")

    st.markdown(
        """
        **Supported documents:**

        - IMP02 — AI Information Compliance Policy
        - IMP03 — User Account Management Policy
        - IMP06 — Systems Administration Policy
        - IMP07 — Data Protection Policy
        - IMP08 — Information Access Control Policy
        - IMP09 — Acceptable Use Policy
        """
    )

    st.markdown("**Processing:** Local computer")

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


st.title("Warwick Policy Assistant")

st.caption(
    "A local RAG research prototype for answering English questions "
    "using selected University of Warwick information management policies."
)

st.info(
    """
    **Scope of this research prototype**

    This chatbot retrieves evidence from six University of Warwick
    information management policies: AI information compliance, user
    account management, systems administration, data protection,
    information access control, and acceptable use.

    It answers English questions using only the content of these six
    documents. Questions about other University policies, services,
    courses, facilities, or general information may not be answerable.
    """
)

st.warning(
    "Research prototype only. Generated answers should be checked "
    "against the displayed policy evidence and the official University "
    "policy documents."
)


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    role = message.get("role", "assistant")

    with st.chat_message(role):
        if role == "assistant":
            render_assistant_message(message)
        else:
            st.markdown(message.get("content", ""))


question = st.chat_input("Ask a policy question in English")


if question:
    user_message = {
        "role": "user",
        "content": question,
    }

    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner(
                "Searching the policy documents and generating an answer..."
            ):
                result = answer_question(question)

            assistant_message = prepare_assistant_message(result)
            render_assistant_message(assistant_message)

        except Exception:
            assistant_message = {
                "role": "assistant",
                "content": (
                    "The local question-answering system is currently "
                    "unavailable. Please confirm that Ollama is running "
                    "and try again."
                ),
                "status": "error",
                "sources": [],
            }

            render_assistant_message(assistant_message)

        st.session_state.messages.append(assistant_message)