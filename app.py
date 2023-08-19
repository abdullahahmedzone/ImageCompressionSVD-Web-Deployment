import streamlit as st
import numpy as np
from PIL import Image


st.title("SVD Image Compresser By Abdullah Ahmed")

# Let User Upload Single Image File Of Types {'jpeg', 'jpg', 'png'}
uploaded_file = st.file_uploader("Upload an image", type=['jpeg', 'jpg', 'png'])


# Checking If A Valid Image File Is Uploaded To Avoid Streamlit Error
if uploaded_file is not None:
    # Open Image File Using Pillow
    image = Image.open(uploaded_file)
    
    # Print The Original Image
    st.image(image, caption = 'Original Image:')
    image = np.array(image)   # Convert Image To Numpy Array 

    for k in range(1, 31):
        # Splitting Color Arrays
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        # Calculating Grayscale Array Using Weighted Equation
        gray_scaled_image = 0.2989 * r + 0.5870 * g + 0.1140 * b
        # Performing Singular Value Decomposition
        U, S, Vt = np.linalg.svd(gray_scaled_image)
        # Compressed Image Array Calculation Using dot Product
        compressed_image = np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))
        # Calculating Compression Ratio
        retained_singular_values = k
        total_singular_values = min(gray_scaled_image.shape)
        compression_ratio = retained_singular_values / total_singular_values

        # Displaying Compressed Images & Compression ratios 
        st.write(f'Compressed Image (k = {k}) (ratio = {compression_ratio:.3f})')
        st.image(compressed_image / 255, clamp = True) # Normalizing Images Array Between 0 , 1
    


















# st.title("SVD Web App By Abdullah Ahmed")

# # Use file uploader to get the uploaded image file
# uploaded_file = st.file_uploader("Upload Image Please", type=['jpg', 'jpeg', 'png'])

# if uploaded_file is not None:
#     # Read the uploaded image using OpenCV
#     image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    
#     if image is not None:
#         # Display the image using Streamlit
#         st.image(image, caption="Uploaded Image")
#     else:
#         st.write("Error reading the uploaded image.")
