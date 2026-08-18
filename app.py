import os
import uuid
import traceback

import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, redirect, url_for, session
from dotenv import load_dotenv
from PIL import Image
from groq import Groq


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

FLASK_SECRET_KEY = os.getenv(
    "FLASK_SECRET_KEY",
    "change-this-secret-key"
)


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

app.secret_key = FLASK_SECRET_KEY

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ============================================================
# GROQ CONFIGURATION
# ============================================================

groq_client = None

if GROQ_API_KEY:

    try:

        groq_client = Groq(
            api_key=GROQ_API_KEY
        )

        print("=" * 60)
        print("Groq API loaded successfully.")
        print("=" * 60)

    except Exception as e:

        print("Groq initialization error:")
        print(e)

else:

    print("=" * 60)
    print("WARNING: GROQ_API_KEY not found.")
    print("=" * 60)


# ============================================================
# LOAD VGG16 MODEL
# ============================================================

MODEL_PATH = "VGG_16_brain_tumor.keras"

print("=" * 60)
print("Loading VGG16 model...")
print("=" * 60)

try:

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("VGG16 model loaded successfully.")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)

except Exception as e:

    print("MODEL LOADING ERROR")
    print(e)

    model = None


# ============================================================
# CLASS NAMES
# IMPORTANT:
# These MUST match the order used during model training.
# ============================================================

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no tumor",
    "pituitary"
]


IMG_SIZE = (224, 224)


# ============================================================
# ALLOWED FILES
# ============================================================

def allowed_file(filename):

    allowed_extensions = {
        "jpg",
        "jpeg",
        "png",
        "bmp"
    }

    if not filename:
        return False

    if "." not in filename:
        return False

    extension = filename.rsplit(
        ".",
        1
    )[1].lower()

    return extension in allowed_extensions


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(filepath):

    image = Image.open(
        filepath
    ).convert("RGB")

    image = image.resize(
        IMG_SIZE
    )

    image = np.asarray(
        image,
        dtype=np.float32
    )

    image = image / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_image(filepath):

    if model is None:

        raise RuntimeError(
            "VGG16 model is not loaded."
        )

    image = preprocess_image(
        filepath
    )

    prediction = model.predict(
        image,
        verbose=0
    )

    probabilities = prediction[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_index] * 100
    )

    if predicted_index < len(CLASS_NAMES):

        predicted_class = (
            CLASS_NAMES[predicted_index]
        )

    else:

        predicted_class = (
            f"class_{predicted_index}"
        )

    return predicted_class, confidence


# ============================================================
# COMMON PAGE DATA
# ============================================================

def get_page_context():

    return {

        "analyzed":
            "prediction" in session,

        "prediction":
            session.get("prediction"),

        "confidence":
            session.get(
                "confidence",
                0
            ),

        "user_question":
            "",

        "ai_answer":
            None,

        "error":
            None,

        "ai_active":
            False
    }


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    context = get_page_context()

    return render_template(
        "index.html",
        **context
    )


# ============================================================
# PREDICT MRI
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # CHECK FILE
        # ----------------------------------------------------

        if "file" not in request.files:

            context = get_page_context()

            context["error"] = (
                "Please select an MRI image."
            )

            return render_template(
                "index.html",
                **context
            )


        file = request.files["file"]


        if file.filename == "":

            context = get_page_context()

            context["error"] = (
                "Please select an MRI image."
            )

            return render_template(
                "index.html",
                **context
            )


        # ----------------------------------------------------
        # CHECK FILE TYPE
        # ----------------------------------------------------

        if not allowed_file(
            file.filename
        ):

            context = get_page_context()

            context["error"] = (
                "Invalid image format. "
                "Use JPG, JPEG, PNG or BMP."
            )

            return render_template(
                "index.html",
                **context
            )


        # ----------------------------------------------------
        # CREATE UNIQUE FILE NAME
        # ----------------------------------------------------

        extension = file.filename.rsplit(
            ".",
            1
        )[1].lower()

        filename = (
            uuid.uuid4().hex
            + "."
            + extension
        )


        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )


        # ----------------------------------------------------
        # SAVE IMAGE
        # ----------------------------------------------------

        file.save(
            filepath
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        predicted_class, confidence = (
            predict_image(filepath)
        )


        print("=" * 60)

        print(
            "Prediction:",
            predicted_class
        )

        print(
            "Confidence:",
            f"{confidence:.2f}%"
        )

        print("=" * 60)


        # ----------------------------------------------------
        # SAVE RESULT IN SESSION
        # ----------------------------------------------------

        session["prediction"] = (
            predicted_class
        )

        session["confidence"] = (
            confidence
        )


        # Remove previous AI conversation

        session.pop(
            "last_question",
            None
        )

        session.pop(
            "last_answer",
            None
        )


        # ----------------------------------------------------
        # DELETE TEMPORARY IMAGE
        # ----------------------------------------------------

        try:

            os.remove(
                filepath
            )

        except Exception:

            pass


        # ----------------------------------------------------
        # RETURN HOME
        # ----------------------------------------------------

        return redirect(
            url_for("home")
        )


    except Exception as e:

        print("=" * 60)
        print("PREDICTION ERROR")
        print("=" * 60)

        traceback.print_exc()

        context = get_page_context()

        context["error"] = (
            "An error occurred while "
            "analyzing the MRI image."
        )

        return render_template(
            "index.html",
            **context
        )


# ============================================================
# ASK GROQ AI
# ============================================================

@app.route(
    "/ask",
    methods=["POST"]
)
def ask():

    question = request.form.get(
        "question",
        ""
    ).strip()


    context = get_page_context()

    context["user_question"] = (
        question
    )

    context["ai_active"] = True


    # --------------------------------------------------------
    # EMPTY QUESTION
    # --------------------------------------------------------

    if not question:

        context["error"] = (
            "Please enter a question."
        )

        return render_template(
            "index.html",
            **context
        )


    # --------------------------------------------------------
    # CHECK GROQ
    # --------------------------------------------------------

    if groq_client is None:

        context["error"] = (
            "Groq API is not configured. "
            "Please check your .env file."
        )

        return render_template(
            "index.html",
            **context
        )


    # --------------------------------------------------------
    # GET CURRENT PREDICTION
    # --------------------------------------------------------

    prediction = session.get(
        "prediction"
    )

    confidence = session.get(
        "confidence",
        0
    )


    if prediction:

        model_context = f"""
The user's MRI was classified by a
deep-learning model as:

Predicted class: {prediction.upper()}
Model confidence: {confidence:.2f}%

This prediction is NOT a confirmed medical diagnosis.
"""

    else:

        model_context = """
There is currently no MRI classification available.
"""


    # --------------------------------------------------------
    # GROQ SYSTEM PROMPT
    # --------------------------------------------------------

    system_prompt = """
You are the AI educational assistant for a
Brain Tumor AI research project.

Your job is to answer the user's exact question.

IMPORTANT:

1. Answer the question directly.
2. Do not force every answer into a fixed structure.
3. Do not automatically provide:
   - What is it?
   - Symptoms
   - Diagnosis
   - Treatment
   unless those topics are relevant to the question.
4. Generate the answer dynamically based on the
   user's actual question.
5. If the question is simple, provide a concise answer.
6. If the question requires explanation, provide
   a detailed but easy-to-understand answer.
7. You may use bullet points or short headings when
   they improve readability.
8. Explain medical terminology in simple language.
9. Never claim that an AI MRI prediction is a confirmed
   medical diagnosis.
10. Do not provide personalized treatment instructions.
11. For medical decisions, advise consultation with a
    qualified healthcare professional.
12. Do not invent patient-specific information.

The user may ask about:
- glioma
- meningioma
- pituitary tumors
- brain tumors
- MRI
- symptoms
- diagnosis
- treatment
- prognosis
- risk factors
- medical terminology
- differences between tumor types
- general brain health

Answer naturally and conversationally.
"""


    # --------------------------------------------------------
    # USER PROMPT
    # --------------------------------------------------------

    user_prompt = f"""
{model_context}

User question:

{question}

Please answer this exact question.
"""


    # --------------------------------------------------------
    # CALL GROQ
    # --------------------------------------------------------

    try:

        response = (
            groq_client
            .chat
            .completions
            .create(

                model="openai/gpt-oss-120b",

                messages=[

                    {
                        "role": "system",
                        "content": system_prompt
                    },

                    {
                        "role": "user",
                        "content": user_prompt
                    }

                ],

                temperature=0.35,

                max_tokens=1200
            )
        )


        answer = (
            response
            .choices[0]
            .message
            .content
        )


        context["ai_answer"] = (
            answer
        )


        # Save last question/answer

        session["last_question"] = (
            question
        )

        session["last_answer"] = (
            answer
        )


        return render_template(
            "index.html",
            **context
        )


    except Exception as e:

        print("=" * 60)
        print("GROQ API ERROR")
        print("=" * 60)

        traceback.print_exc()

        context["error"] = (
            "Unable to generate the AI response. "
            "Please try again."
        )

        return render_template(
            "index.html",
            **context
        )


# ============================================================
# CLEAR RESULT
# ============================================================

@app.route("/clear")
def clear():

    session.clear()

    return redirect(
        url_for("home")
    )


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("BRAIN TUMOR AI")
    print("=" * 60)

    print(
        "Open: http://127.0.0.1:5000"
    )

    print("=" * 60)

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )