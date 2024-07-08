import time #Iwish
import os
import json
import requests
import streamlit as st
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import google.generativeai as genai


def main():
    # Set page configuration
    st.set_page_config(
        page_title="Alwrity - AI YouTube Title Generator",
        layout="wide",
    )
    # Remove the extra spaces from margin top.
    st.markdown("""
        <style>
        ::-webkit-scrollbar-track {
        background: #e1ebf9;
        }

        ::-webkit-scrollbar-thumb {
            background-color: #90CAF9;
            border-radius: 10px;
            border: 3px solid #e1ebf9;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #64B5F6;
        }

        ::-webkit-scrollbar {
            width: 16px;
        }
        div.stButton > button:first-child {
            background: #1565C0;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 10px 2px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
            font-weight: bold;
        }
        </style>
    """
    , unsafe_allow_html=True)

    # Hide top header line
    hide_decoration_bar_style = '<style>header {visibility: hidden;}</style>'
    st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

    # Hide footer
    hide_streamlit_footer = '<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>'
    st.markdown(hide_streamlit_footer, unsafe_allow_html=True)

    st.title("✍️  Alwrity - AI YouTube Video Title Generator")

    # Input section
    with st.expander("**PRO-TIP 💡** - Read the instructions below.", expanded=True):
        col1, col2 = st.columns([5, 5])
        with col1:
            main_points = st.text_area(
                '**What is your video about?**',
                placeholder='Write a few words about your video for the title (e.g., "New trek, Latest in news, Finance, Tech...")'
            )
            tone_style = st.selectbox(
                '**Select Tone & Style 📝**',
                ['Casual', 'Professional', 'Humorous', 'Formal', 'Informal', 'Inspirational']
            )
        with col2:
            target_audience = st.multiselect(
                '**Select Video Target Audience (One or Multiple) 👥**',
                [
                    'Beginners', 'Marketers', 'Gamers', 'Foodies', 'Entrepreneurs',
                    'Students', 'Parents', 'Tech Enthusiasts', 'General Audience', 
                    'News article', 'Finance Article'
                ]
            )
            use_case = st.selectbox(
                '**YouTube Title Use Case 📹**',
                [
                    'Tutorials', 'Product Reviews', 'Explainer Videos', 'Vlogs',
                    'Motivational Speeches', 'Comedy Skits', 'Educational Content'
                ]
            )
    
    # Generate YouTube Titles button
    if st.button('**Generate YouTube Titles 🎬**'):
        with st.status("Assigning AI professional to write your YT Titles..", expanded=True) as status:
            if not main_points:
                st.error("🚫 Please provide all required inputs.")
            else:
                response = generate_youtube_title(target_audience, main_points, tone_style, use_case)
                if response:
                    st.subheader(f'**🧕👩: Your Final youtube Titles !**')
                    st.markdown(response)
                    st.write("\n\n\n\n\n\n")
                else:
                    st.error("💥**Failed to write Letter. Please try again!**")


def generate_youtube_title(target_audience, main_points, tone_style, use_case):
    """ Generate youtube script generator """

    prompt = f"""
    **Instructions:**
    
    Please generate 5 YouTube title options for a video about **{main_points}** based on the following information:
    
    
    **Target Audience:** {target_audience}
    
    **Tone and Style:** {tone_style}
    
    **Use Case:** {use_case}
    
    **Specific Instructions:**
    
    * Make the titles catchy and attention-grabbing.
    * Use relevant keywords to improve SEO.
    * Tailor the language and tone to the target audience.
    * Ensure the title reflects the content and use case of the video.
    """

    try:
        response = generate_text_with_exception_handling(prompt)
        return response
    except Exception as err:
        st.error(f"Exit: Failed to get response from LLM: {err}")
        exit(1)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_text_with_exception_handling(prompt):
    """
    Generates text using the Gemini model with exception handling.

    Args:
        api_key (str): Your Google Generative AI API key.
        prompt (str): The prompt for text generation.

    Returns:
        str: The generated text.
    """

    try:
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

        generation_config = {
            "temperature": 1,
            "top_p": 0.7,
            "top_k": 0,
            "max_output_tokens": 8192,
        }

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]

        model = genai.GenerativeModel(model_name="gemini-1.5-flash",
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)

        convo = model.start_chat(history=[])
        convo.send_message(prompt)
        return convo.last.text

    except Exception as e:
        st.exception(f"An unexpected error occurred: {e}")
        return None


if __name__ == "__main__":
    main()
