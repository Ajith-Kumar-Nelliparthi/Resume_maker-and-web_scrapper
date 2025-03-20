# Resume Maker and Web Scrapper

This project combines a **Resume Maker** and a **Web Scrapper** to assist users in creating professional resumes and extracting relevant data from the web for research or analysis.

## Features

### Resume Maker
- Generate professional resumes in various formats.
- Analyze resumes for keyword optimization using the `resume_analyzer_debug.py` module.
- Integrates with word processing tools for customization.

### Web Scrapper
- Scrape data from websites using the `web_scrapper.py` module.
- Extract entities and save structured data in JSON format.
- Includes pre-scraped data for reference (`scraped_data.json`).

## Project Structure
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


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Resume_maker-and-web_scrapper.git

2. Navigate to the project directory:
    cd Resume_maker-and-web_scrapper

3. Install dependencies
    ```bash
    pip install -r requirements.txt

## Usage

Resume Maker
Configure the .env file in the resume_maker directory with necessary settings.
Run the resume_analyzer_debug.py or res.py script to create or analyze resumes:
    ```bash
    python resume_maker/resume_analyzer_debug.py

Web Scrapper
Use the web_scrapper.py script to scrape data:
``` python web_scrapper/web_scrapper.py ```
Extracted data will be saved in scraped_data.json.

## Dependencies

Python 3.10 or higher
Required libraries are listed in requirements.txt.

## License
This project is licensed under the MIT License.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## Contact
For any inquiries, please contact nelliparthi123@gmail.com. ```