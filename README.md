# RetroBoard: Computer Vision & AI Recommendation Engine

A full-stack web application that processes uploaded images through custom-built mathematical color-grading pipelines and utilizes a vector database to perform nearest-neighbor similarity searches. 

This project goes beyond basic preset filters by utilizing NumPy and OpenCV to manipulate pixel arrays, map non-linear S-curves, and generate optical diffusion, mimicking the physical chemistry of vintage analog film.

## 🚀 Technical Features

* **Custom 1D Lookup Tables (LUTs):** Maps non-linear RGB curves to create professional-grade, cinematic color grading without relying on external preset files.
* **Algorithmic Optical Bloom:** Isolates image highlights using binary thresholding and applies heavy Gaussian blur to create a high-end, diffused lens glow.
* **Vector Embeddings & Similarity Search:** Extracts the dominant color signature from processed images, converts it to a 3-dimensional floating-point vector, and queries a FAISS index to return aesthetically similar images in real-time.
* **Optimized Matrix Operations:** Uses NumPy for vectorized mathematical operations on image arrays, dramatically reducing processing time compared to standard iterative loops.
* **Secure File Handling:** Implements UUID generation for file storage to prevent data collisions and ensure secure, concurrent user uploads.

## 🛠️ Tech Stack

* **Backend:** Python, Flask
* **Computer Vision & Math:** OpenCV (`cv2`), NumPy
* **Vector Database (AI):** FAISS (Facebook AI Similarity Search)
* **Frontend:** HTML5, CSS3, Jinja2 Templating

## 🧠 How the Filters Work Under the Hood

The application includes four distinct processing pipelines:
1.  **Cinematic Vintage Cam:** Combines an optical bloom algorithm with custom Teal & Orange S-curves. Lifts absolute blacks to create a matte paper effect.
2.  **Golden Hour Warmth:** Matrix multiplication to boost red/green channels while suppressing blues, combined with a lifted shadow baseline.
3.  **90s Retro Cross-Process:** Manipulates the HSV color space to desaturate the image while pushing magenta into the shadows and yellow into the highlights.
4.  **Classic B&W Grunge:** Converts to grayscale, applies a high-alpha contrast scaler, and injects heavy, randomized Gaussian noise to mimic high-ISO film grain.

## 💻 Local Setup & Installation

To run this project on your local machine:

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME
