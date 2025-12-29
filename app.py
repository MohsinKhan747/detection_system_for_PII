import streamlit as st


from pii_detector import analyze_text


def main() -> None:
    st.set_page_config(
        page_title="PII Detection Demo",
        page_icon="🔒",
        layout="wide",
    )

    st.title("PII Detection")

    with st.sidebar:
        st.header("Settings")
        show_highlight = st.checkbox("Highlight PII in text", value=True)


    text = st.text_area(
        "Input text",
        height=220,
        help="Your text never leaves this session; everything runs locally.",
    )

    if not text.strip():
        st.info("Enter some text above to run PII detection.")
        return

    # Convert to uppercase before detection as requested
    processed_text = text.upper()

    # Highlight sensitive fields including names (but not generic locations/orgs)
    sensitive_labels = [
        "NAME",
        "PASSWORD",
        "EMAIL",
        "PHONE",
        "IP_ADDRESS",
        "NATIONAL_ID",
        "CREDIT_CARD",
        "BANK_ACCOUNT",
    ]

    with st.spinner("Analyzing text for PII..."):
        result = analyze_text(processed_text, enabled_labels=sensitive_labels)

    # Layout: left = highlighted text, right = grouped entities
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Original Text")
        if show_highlight:
            st.markdown(result.highlighted_markdown, unsafe_allow_html=True)
        else:
            st.code(processed_text)



if __name__ == "__main__":
    main()
