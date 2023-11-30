import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.feature_extraction import image
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jaccard



def load_image(file):
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def compare_histograms(img1, img2):
    # Calculate histogram for each image and normalize
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    # Compare using correlation
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity

def compare_ssim(img1, img2):
    # Convert images to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    # Compute SSIM
    score, _ = ssim(img1_gray, img2_gray, full=True)
    return score

def compare_features(img1, img2):
    # Feature matching (simplified)
    # Initialize the ORB detector
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # Create BFMatcher and match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    # Sort matches based on distance
    matches = sorted(matches, key=lambda x: x.distance)
    # Calculate the similarity based on the top N matches
    similarity = sum([1 - m.distance/100 for m in matches[:30]]) / 30
    return similarity

def compare_jaccard(img1, img2):
    # Flatten the images to 1D arrays
    img1_flat = img1.ravel()
    img2_flat = img2.ravel()
    # Compute Jaccard similarity
    return 1 - jaccard(img1_flat, img2_flat)

def compare_cosine(img1, img2):
    # Flatten the images to 1D arrays
    img1_flat = img1.ravel().reshape(1, -1)
    img2_flat = img2.ravel().reshape(1, -1)
    # Compute cosine similarity
    return cosine_similarity(img1_flat, img2_flat)[0][0]



st.title("Image Similarity Checker")

# Add a camera input
camera_img = st.camera_input("Take a picture")

# Add a file uploader
uploaded_img = st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg"])

user_img = None

# Process the image (either from camera or upload)
if camera_img is not None:
    user_img = load_image(camera_img)
elif uploaded_img is not None:
    user_img = load_image(uploaded_img)

if user_img is not None:

    st.image(user_img, caption='Your Image', use_column_width=True)

    # Load the three given images
    img1 = load_image(open("clarkson.png", "rb"))
    img2 = load_image(open("hammond.png", "rb"))
    img3 = load_image(open("may.png", "rb"))

    images = [img1,img2,img3]

    # Display the given images

    for col, img in zip(st.columns(3), images):
        col.image(img, caption=f'Image', use_column_width=True)

    algorithm = st.selectbox("Choose an algorithm", ["Histogram", "Feature Matching"])#, "SSI", "Jaccard", "Cosine"

    if st.button('Compare'):
        # Perform comparison based on the selected algorithm
        comparison_functions = {
            "Histogram": compare_histograms,
            "SSI": compare_ssim,
            "Feature Matching": compare_features,
            "Jaccard": compare_jaccard,
            "Cosine": compare_cosine
        }

        # Compute similarity scores
        scores = [comparison_functions[algorithm](user_img, img) for img in [img1, img2, img3]]

        # Find the index of the most similar image
        most_similar_index = scores.index(max(scores))

        # Display the most similar image
        st.image([img1, img2, img3][most_similar_index], caption=f'Most Similar Image (Image {most_similar_index + 1})', use_column_width=True)