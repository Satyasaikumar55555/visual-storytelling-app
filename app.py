import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, GPT2LMHeadModel, GPT2Tokenizer

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")

st.title("The Art of Visual Storytelling")

genre_options = ["Adventure", "Fantasy", "Mystery", "Romance", "Horror", "Sci-Fi"]
selected_genre = st.selectbox("Choose a story genre", genre_options)

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)

        with st.spinner("Generating caption..."):
            inputs = processor(image, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
        st.write(f"Caption: {caption}.init()")

        if st.button('Generate Story from Caption'):
            with st.spinner("Generating story..."):
                genre_prompt = f"The {selected_genre.lower()} story based on the following description: {caption} \n"
                input_ids = tokenizer_gpt2.encode(genre_prompt, return_tensors='pt', max_length=1024, truncation=True)
                story = model_gpt2.generate(
                    input_ids,
                    max_length=200,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    temperature=0.9,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=tokenizer_gpt2.eos_token_id
                )
                story_text = tokenizer_gpt2.decode(story[0], skip_special_tokens=True)
            st.text_area(f"Generated {selected_genre} Story", story_text, height=250)

            story_filename = f"{selected_genre}_story.txt"
            st.download_button(
                label="Download Story",
                data=story_text,
                file_name=story_filename,
                mime="text/plain"
            )
    except Exception as e:
        st.error(f"Error: {e}")
