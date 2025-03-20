# Resume Maker and Web Scrapper

This project combines a **Resume Maker** and a **Web Scrapper** to assist users in creating professional resumes and extracting relevant data from the web for research or analysis.

## Overview
Resume Analyzer is a Python-based tool designed to help users optimize their resumes for Applicant Tracking Systems (ATS). It analyzes an existing resume against a job description, provides insights such as ATS compatibility scores and missing skills, and generates an improved resume tailored to the job. The tool outputs the rewritten resume in both a text file (rewritten_resume.txt) and a Microsoft Word document (rewritten_resume.docx), with section headers (e.g., "Summary", "Work Experience") followed by horizontal lines for better readability. Users can review and save the Word document manually.
This project leverages natural language processing (NLP), machine learning, and the Grok API from xAI to enhance resume content, making it a valuable asset for job seekers aiming to stand out.

## Features

### Resume Maker
- Resume Analysis: Calculates an ATS score, identifies missing skills, and checks degree relevance.
- Skill Extraction: Extracts technical and soft skills from both the resume and job description using NLP and the Grok API.
- Resume Rewriting: Generates an ATS-optimized resume with relevant keywords and professional formatting.
Output Formats:
- Saves a plain text version (rewritten_resume.txt).
- Creates and opens a Word document (rewritten_resume.docx) with formatted sections and horizontal lines after headers.
- Interactive Chat: Provides a chat interface for additional resume-related advice.



### Web Scrapper
- Scrape data from websites using the `web_scrapper.py` module.
- Extract entities and save structured data in JSON format.
- Includes pre-scraped data for reference (`scraped_data.json`).

## Project Structure
```
Resume_maker-and-web_scrapper/ 
├── .gitignore 
├── LICENSE 
├── README.md 
├── requirements.txt 
├── research/ │ └── web_scrapping.ipynb 
├── resume_maker/ 
│ ├── init.py 
│ ├── .env 
│ ├── res.py 
│ ├── resume_analyzer_debug.py 
│ ├── resume1.py 
│ ├── word.py 
│ ├── word1.py 
│ └── pycache/ 
├── web_scrapper/ 
│ ├── init.py 
│ ├── Allen_Newell.json 
│ ├── entity.py 
│ ├── scraped_data.json 
│ ├── web_scrapper.py 
│ └── pycache/
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ajith-Kumar-Nelliparthi/Resume_maker-and-web_scrapper.git

2. Navigate to the project directory:
```
    cd resume_maker
```

4. Install dependencies
    ```bash
    pip install -r requirements.txt
5.  Set up [Grok API](https://x.ai/) key:\
Obtain an API key from Grok.\
Store it in an environment variable or a .env file.

## Usage

### Resume Maker
1. Run the Script:
```
python word.py
```
2. Provide Inputs:
- Resume File: Enter the path to your resume (.pdf, .docx, or .txt).
-  Job Title: Specify the job title you’re targeting (e.g., "Data Scientist").
-  job Description: Paste the job description, ending with "END" on a new line.
3. Review Insights:
The script outputs analysis insights (ATS score, missing skills, etc.) to the console.
4. Chat Interface:
Interact with the assistant for additional advice (type "exit" to proceed).
5. Output:
- Text File: rewritten_resume.txt is saved in the project directory.
- Word Document: rewritten_resume.docx is created and opened in Microsoft Word with formatted sections and horizontal lines (underscores) after headers. Review and save it manually.

### Web Scrapper
Use the web_scrapper.py script to scrape data:\
``` python web_scrapper/web_scrapper.py ```
Extracted data will be saved in scraped_data.json.

## Example
![Screenshot 2025-03-20 111508](https://github.com/user-attachments/assets/50b52d73-34ac-418d-9b0c-3bd300129cc3)
![Screenshot 2025-03-20 111526](https://github.com/user-attachments/assets/8b6983cc-ff1a-4ac3-88a7-b02e675c7392)
![Screenshot 2025-03-20 111550](https://github.com/user-attachments/assets/83747f40-55a5-47f1-8f9b-c45d868dcfb4)
![Screenshot 2025-03-20 111611](https://github.com/user-attachments/assets/820c5ed8-c2d9-47c6-949e-3ad39d84e6bd)


## Dependencies

Python 3.10 or higher\
Microsoft Word: Required to open and edit the ```.docx``` output (or a compatible application like LibreOffice).\
Required libraries are listed in requirements.txt.

## Troubleshooting
1. Error: "Error creating or opening Word document":
- Ensure Microsoft Word (or a .docx viewer) is installed.
- Check if python-docx is installed (pip install python-docx).

2. API Issues:
- Confirm your Grok API key is valid and correctly set in .env.

3. Missing Content:
- Check the input resume file for readability (e.g., avoid locked PDFs).
For additional help, review the console output for specific error messages and report them.

## License
This project is licensed under the [MIT License](#license).

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## Acknowledgments
- Uses python-docx for Word document generation.
- Inspired by the need to simplify job application success.

## Contact
For any inquiries, please contact nelliparthi123@gmail.com.
