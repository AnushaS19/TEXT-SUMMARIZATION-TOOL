# TEXT-SUMMARIZATION-TOOL

COMPANY : ``CODTECH IT SOLUTIONS 

NAME : ANUSHA S

INTERN ID :CT04DH1155

DOMAIN : ARTIFICIAL INTELLIGENCE

DURATION  : 4 WEEKS

MENTOR  : NEELA SANTHOSH

DESCRIPTION : This project is a comprehensive, interactive, and smart Text Summarizer Web Application designed to help users simplify long passages of text into short, meaningful summaries. Built entirely with Python, it leverages powerful libraries and frameworks such as Streamlit, NLTK, NetworkX, NumPy, and gTTS to deliver a seamless, user-friendly experience. This app is ideal for students, researchers, writers, and professionals who work with large volumes of textual data and need a fast, reliable tool to extract essential information.

🧠 Key Technologies and Libraries Used:
Python: The core programming language used for backend logic and data processing.

.Streamlit: A fast, interactive, and lightweight framework for building data apps and web interfaces directly in Python.

.NLTK (Natural Language Toolkit): Used for natural language processing tasks such as tokenization, text cleaning, and stopword removal.

.NumPy: Utilized for efficient numerical computations, especially while working with similarity matrices.

.NetworkX: Helps build a graph of sentence similarity and apply the PageRank algorithm to rank sentences.

.gTTS (Google Text-to-Speech): Converts generated summary text into spoken audio using Google’s speech synthesis engine.

.tempfile: A built-in module used to handle temporary storage of audio files for playback in Streamlit apps.

.Base64: Used for encoding the generated summary so that users can download it directly from the app as a .txt file.

⚙️ How It Works:
.Input Text:
The app allows users to either upload a .txt file or paste their own text into a provided text area. This flexibility supports both document summarization and quick article processing.

.Preprocessing:
The input text is cleaned by removing punctuation, digits, and converting everything to lowercase. It is then tokenized into words and filtered through English stopwords using NLTK to ensure meaningful sentence comparison.

.Sentence Similarity Calculation:
Every sentence is compared with every other sentence to compute a cosine similarity score based on word frequency vectors. This is stored in a similarity matrix using NumPy.

.Graph Creation and Ranking:
The similarity matrix is converted into a graph where sentences are nodes and edges represent similarity scores. The PageRank algorithm (via NetworkX) is used to identify the most important sentences in the document.

.Summary Generation:
Based on the PageRank scores, the top N sentences are selected (user-defined) and presented in the user's preferred format: paragraph, bullet points, or numbered list.

.Audio Output:
Using gTTS, the final summary is converted into speech. The audio is temporarily saved using Python’s tempfile module and streamed directly in the browser using Streamlit’s audio widget. Users can also choose the language for speech output (en for English, hi for Hindi, kn for Kannada).

.Text Download and Statistics:
The app also provides a direct link to download the summary as a .txt file. Word and character counts are displayed to compare the original and summarized versions.

🌟 Features Highlight:
✅ Accepts both file uploads and manual text input.

✅ Adjustable number of summary sentences using a slider.

✅ Choose from paragraph, bullet, or numbered summary format.

✅ Generate and play audio summary in multiple languages.

✅ Downloadable .txt file of the summary.

✅ Live word and character count for both original and summary.

✅ Stylish user interface with date display and a custom background.

✅ No external frontend or JavaScript needed — everything handled within Python.

💡 Why This Project?
This project showcases the practical combination of Natural Language Processing, graph algorithms, and text-to-speech features in a single web application. It's especially helpful for simplifying academic papers, blog articles, or reports, and making summaries more accessible via audio—useful for the visually impaired or multitasking users.

Whether you're a developer learning NLP, a student who wants to digest long chapters faster, or a productivity enthusiast, this project serves as a robust, real-world example of how Python and open-source libraries can come together to solve everyday problems.


OUTPUT ---:

![Image](https://github.com/user-attachments/assets/7d1d080a-aecc-43cd-a5a7-7791fc4c17db)
