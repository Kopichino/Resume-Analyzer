from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF
import spacy
from sentence_transformers import SentenceTransformer, util
from typing import List
import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# Initialize FastAPI app
app = FastAPI(title="Smart Resume Analyzer API")

# Load spaCy model for NLP
nlp = spacy.load("en_core_web_sm")

# Load SentenceTransformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample skills list for matching (extend this as needed)
SKILLS_DB = [
    "python", "java", "javascript", "sql", "machine learning", "data analysis",
    "project management", "aws", "docker", "react", "node.js", "typescript",
    "database management", "cloud computing", "api development"
]

def extract_text_from_pdf(file: UploadFile) -> str:
    """Extract text from a PDF resume."""
    try:
        pdf_document = fitz.open(stream=file.file.read(), filetype="pdf")
        text = ""
        for page in pdf_document:
            text += page.get_text()
        pdf_document.close()
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")

def extract_skills(text: str) -> List[str]:
    """Extract skills from resume text using spaCy and a predefined skills list."""
    doc = nlp(text.lower())
    found_skills = []
    for skill in SKILLS_DB:
        if skill in text.lower():
            found_skills.append(skill)
    return list(set(found_skills))

def calculate_keyword_density(text: str, keywords: List[str]) -> float:
    """Calculate keyword density in the resume."""
    tokens = word_tokenize(text.lower())
    keyword_count = sum(1 for token in tokens if token in [k.lower() for k in keywords])
    total_words = len(tokens)
    return (keyword_count / total_words * 100) if total_words > 0 else 0

def generate_suggestions(resume_skills: List[str], job_skills: List[str]) -> List[str]:
    """Generate suggestions for missing skills."""
    missing_skills = [skill for skill in job_skills if skill not in resume_skills]
    suggestions = [f"Consider adding experience or certification in '{skill}' to strengthen your resume." for skill in missing_skills]
    return suggestions

@app.post("/analyze-resume/")
async def analyze_resume(resume: UploadFile = File(...), job_description: str = ""):
    """Analyze a resume against a job description."""
    if not resume.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Extract text from resume
    resume_text = extract_text_from_pdf(resume)

    # Extract skills from resume and job description
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)

    # Calculate skills match score
    skills_match = len(set(resume_skills) & set(job_skills)) / len(set(job_skills)) * 100 if job_skills else 0

    # Calculate keyword density
    job_keywords = job_skills + re.findall(r'\b\w+\b', job_description.lower())
    keyword_density = calculate_keyword_density(resume_text, job_keywords)

    # Calculate similarity using SentenceTransformers
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    job_embedding = model.encode(job_description, convert_to_tensor=True)
    similarity_score = util.cos_sim(resume_embedding, job_embedding)[0][0].item() * 100

    # Generate suggestions
    suggestions = generate_suggestions(resume_skills, job_skills)

    # Combine scores (weighted average for simplicity)
    final_score = (skills_match * 0.4 + keyword_density * 0.3 + similarity_score * 0.3)

    return JSONResponse({
        "skills_match_score": round(skills_match, 2),
        "keyword_density_score": round(keyword_density, 2),
        "similarity_score": round(similarity_score, 2),
        "final_score": round(final_score, 2),
        "extracted_skills": resume_skills,
        "suggestions": suggestions
    })

@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"message": "Smart Resume Analyzer API is running!"}