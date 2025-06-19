import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import tempfile

def greeting(name):
    return f"Hello from module2, {name}!"

# Function to process the image, classify it, and crop if clear
def process_image(file_path):
    model = YOLO("best.pt", "v8")

    # Predict with the model
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image:
        temp_image.close()
        Image.open(file_path).save(temp_image.name)
        results = model.predict(source=temp_image.name, conf=0.4, save=True)

    blur_conf_threshold = 0.5
    clear_conf_threshold = 0.9

    # Initialize flags
    is_blur = False
    is_clear = False
    cropped_image = None

    # Process results
    for result in results[0].boxes:
        confidence = result.conf[0].item()  # Extract the confidence score
        if blur_conf_threshold <= confidence <= clear_conf_threshold:
            is_blur = True
        elif confidence > clear_conf_threshold:
            is_clear = True
            box = result.xyxy[0].tolist()  # Extract bounding box coordinates
            cropped_image = crop_image(file_path, box)

    # Return classification and cropped image
    if is_blur:
        return 'The image is blurry. Please reupload the image again!', None
    elif is_clear:
        return 'The image is clear', cropped_image
    else:
        s=("Not Detected! The image is uncertain. Please reupload the image again!")
        return s, None

# Function to crop the image based on bounding box
def crop_image(file_path, box):
    image = Image.open(file_path)
    cropped_image = image.crop(box)
    return cropped_image

def greeting(name):
    return f"Hello module2, {name}!"
def newmethod():
    return "Hello world"
# Streamlit app
def main():
    st.title('Welcome to my AI project')
    st.title('Document Detection')
    st.text('This is a web app to:\n1- Detect documents\n2- Classify if document is clear or blurry\n3- Crop the document image!')

    # File uploader
    uploaded_file = st.file_uploader('Upload your image here:', type=['png', 'jpg', 'jpeg'])
   
    

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.success("Photo uploaded successfully!")

        # Process the image and get classification and cropped image
        classification, cropped_image = process_image(uploaded_file)

        # Display classification result
        st.write('Classification result:', classification)

        # Display cropped image if classification is clear
        if cropped_image is not None:
            st.image(cropped_image, caption='Cropped Document Image.', use_column_width=True)

if __name__ == "__main__":
    main()

