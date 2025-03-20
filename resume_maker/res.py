import os
import json
import re
import PyPDF2
import logging
from dotenv import load_dotenv
from fpdf import FPDF
import groq
import nltk
from nltk.corpus import stopwords

# ✅ Load NLTK stopwords
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))



# ✅ Load API Key from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing API Key! Set GROQ_API_KEY in your environment.")

# ✅ Initialize Groq Client
client = groq.Groq(api_key=GROQ_API_KEY)

# ✅ Function to Generate Resume using Groq API
def generate_resume(user_info):
    prompt = f"Create a resume for the following information: {user_info}"

    try:
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error generating resume: {e}")
        return None

# ✅ Function to Load Resume Content (PDF or TXT)
def load_resume_content(resume_path):
    try:
        if resume_path.endswith('.pdf'):
            with open(resume_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
                return text
        else:
            with open(resume_path, "r", encoding="utf-8") as file:
                return file.read()
    except Exception as e:
        logging.error(f"Error loading resume: {e}")
        return ""

# ✅ Function to Extract Skills from Text
def extract_skills(text):
    words = set(re.findall(r'\b\w+\b', text.lower()))
    filtered_words = words - STOPWORDS  # Remove common words
    return filtered_words

# ✅ Function to Find Missing Skills
def find_missing_skills(resume_content, job_description):
    resume_skills = extract_skills(resume_content)
    job_skills = extract_skills(job_description)
    missing_skills = job_skills - resume_skills
    return list(missing_skills)

# ✅ Function to Calculate ATS Score
def calculate_ats_score(resume_content, job_description):
    resume_words = extract_skills(resume_content)
    job_words = extract_skills(job_description)
    matching_words = resume_words.intersection(job_words)
    ats_score = len(matching_words) / len(job_words) * 100 if job_words else 0
    return round(ats_score, 2)

# ✅ Function to Check Job Title Match
def check_job_title_match(resume_content, job_description):
    job_title = re.search(r'(?<=Title: )\w+', job_description)
    return job_title.group(0) in resume_content if job_title else False

# ✅ Function to Check Degree Match
def check_degree_match(resume_content, job_description):
    degree_keywords = ['Bachelor', 'Master', 'PhD']
    return any(degree in resume_content for degree in degree_keywords)

# ✅ Function to Extract Accomplishments
def extract_accomplishments(resume_content):
    accomplishments = re.findall(r'(?<=Accomplishments: ).*', resume_content)
    return accomplishments if accomplishments else []

# ✅ Function to Rewrite Resume using AI
def rewrite_resume(resume_content, job_description):
    prompt = f"""
    Rewrite the following resume to better match the job description while keeping it ATS-friendly.

    Resume Content:
    {resume_content}

    Job Description:
    {job_description}

    Ensure the rewritten resume:
    - Uses relevant keywords from the job description
    - Highlights missing skills naturally
    - Uses professional and concise language
    - Maintains proper resume formatting

    Provide the improved resume in a structured format.
    """
    try:
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content  
    except Exception as e:
        logging.error(f"Error generating rewritten resume: {e}")
        return "Error generating rewritten resume."

# ✅ Function to Analyze Resume & Generate Insights
def analyze_resume_details(resume_content, job_description):
    insights = {
        "ats_score": calculate_ats_score(resume_content, job_description),
        "missing_skills": find_missing_skills(resume_content, job_description),
        "job_title_match": check_job_title_match(resume_content, job_description),
        "degree_match": check_degree_match(resume_content, job_description),
        "word_count": len(resume_content.split()),
        "accomplishments": extract_accomplishments(resume_content),
        "rewritten_resume": rewrite_resume(resume_content, job_description)
    }
    return insights

# ✅ Function to Save Rewritten Resume (TXT)
def save_rewritten_resume(content, output_path="rewritten_resume.txt"):
    try:
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(content)
        logging.info(f"✅ Rewritten resume saved to: {output_path}")
    except Exception as e:
        logging.error(f"Error saving rewritten resume: {e}")

# ✅ Function to Save Rewritten Resume as PDF
def save_rewritten_resume_pdf(content, output_path="rewritten_resume.pdf"):
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        for line in content.split("\n"):
            pdf.cell(200, 10, txt=line.encode('latin-1', 'replace').decode('latin-1'), ln=True)

        pdf.output(output_path)
        logging.info(f"✅ Rewritten resume saved as PDF: {output_path}")
    except Exception as e:
        logging.error(f"Error saving resume as PDF: {e}")

# ✅ Main Function (User Input & Execution)
def main():
    resume_path = input("Enter the path to your resume: ")
    print("Enter the job description you want to apply for (press Enter twice to finish):")
    
    # Read multi-line job description input
    job_lines = []
    while True:
        line = input()
        if not line.strip():
            break
        job_lines.append(line)
    job_description = "\n".join(job_lines)

    # Load the resume content
    resume_content = load_resume_content(resume_path)
    
    if not resume_content:
        logging.error("❌ No resume content found! Exiting.")
        return

    # Analyze resume against job description
    insights = analyze_resume_details(resume_content, job_description)
    
    # Display insights
    print("\n📊 **Resume Analysis Insights**:")
    print(json.dumps(insights, indent=4))

    # Save rewritten resume
    save_rewritten_resume(insights["rewritten_resume"])
    save_rewritten_resume_pdf(insights["rewritten_resume"])

if __name__ == "__main__":
    main()

