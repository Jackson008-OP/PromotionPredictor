import re
import pdfplumber


DEPARTMENT_KEYWORDS = {
    "Sales & Marketing": ["sales", "marketing", "business development", "crm", "brand"],
    "Technology": ["software", "developer", "engineer", "data", "it ", "tech", "cloud", "devops", "python", "java", "machine learning", "ai "],
    "Operations": ["operations", "logistics", "supply chain", "procurement", "warehouse"],
    "HR": ["human resources", "hr ", "recruitment", "talent", "people"],
    "Finance": ["finance", "accounting", "financial", "audit", "tax", "budget"],
    "Analytics & Reporting": ["analyst", "analytics", "reporting", "bi ", "tableau", "power bi"],
    "Procurement": ["procurement", "vendor", "purchasing", "sourcing"],
    "Legal": ["legal", "compliance", "law", "attorney", "counsel"],
    "R&D": ["research", "r&d", "innovation", "scientist", "lab"],
}

EDUCATION_KEYWORDS = {
    3: ["phd", "ph.d", "doctorate", "master", "mba", "m.tech", "m.sc", "msc", "m.e", "pg diploma"],
    2: ["bachelor", "b.tech", "b.e", "b.sc", "bsc", "b.com", "bca", "b.a", "undergraduate", "degree"],
    1: ["12th", "hsc", "sslc", "10th", "secondary", "diploma", "iti"],
}

AWARD_KEYWORDS = [
    "award", "best employee", "recognition", "outstanding", "excellence",
    "achiever", "performer", "certified", "medal", "honour", "star employee",
]

TRAINING_KEYWORDS = [
    "certified", "certification", "course", "training", "workshop",
    "bootcamp", "program", "completion", "attended", "completed",
]


def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.lower()


def extract_age(text):
    patterns = [
        r"age[:\s]+(\d{2})",
        r"(\d{2})\s*years?\s*old",
        r"dob[:\s]+\d{1,2}[\/\-]\d{1,2}[\/\-](\d{4})",
        r"date of birth[:\s]+\d{1,2}[\/\-]\d{1,2}[\/\-](\d{4})",
        r"born\s+in\s+(\d{4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            val = int(match.group(1))
            if val > 1950:
                return 2025 - val
            if 20 <= val <= 60:
                return val
    return None


def extract_experience(text):
    patterns = [
        r"(\d+)\+?\s*years?\s*(of\s*)?(experience|exp)",
        r"experience[:\s]+(\d+)\s*years?",
        r"(\d+)\s*years?\s*in\s*(the\s*)?(industry|field|domain|sector)",
        r"total\s*experience[:\s]+(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            val = int(match.group(1))
            if 0 < val < 50:
                return val
    return None


def extract_education(text):
    for level, keywords in EDUCATION_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return level
    return 2


def extract_department(text):
    for dept, keywords in DEPARTMENT_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return dept
    return "Technology"


def extract_trainings(text):
    count = 0
    for kw in TRAINING_KEYWORDS:
        count += len(re.findall(rf"\b{kw}\b", text))
    return min(max(1, count // 2), 10)


def extract_avg_training_score(text):
    patterns = [
        r"score[:\s]+(\d{2,3})[\/\s]*100",
        r"(\d{2,3})\s*\/\s*100",
        r"percentage[:\s]+(\d{2,3})",
        r"cgpa[:\s]+(\d+\.?\d*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            val = float(match.group(1))
            if val <= 10:
                return int((val / 10) * 100)
            if 40 <= val <= 100:
                return int(val)
    skill_words = ["python", "sql", "java", "excel", "machine learning", "deep learning",
                   "aws", "azure", "docker", "kubernetes", "tableau", "spark"]
    skill_count = sum(1 for s in skill_words if s in text)
    return min(50 + skill_count * 5, 95)


def extract_awards(text):
    for kw in AWARD_KEYWORDS:
        if kw in text:
            return 1
    return 0


def extract_gender(text):
    if any(w in text for w in ["she ", "her ", "hers ", "ms.", "mrs.", "miss "]):
        return "f"
    if any(w in text for w in ["he ", "his ", "him ", "mr. ", "sir "]):
        return "m"
    return "m"


def extract_previous_rating(text):
    patterns = [
        r"rating[:\s]+(\d\.?\d?)[\/\s]*5",
        r"performance[:\s]+(\d\.?\d?)",
        r"appraisal[:\s]+(\d\.?\d?)",
        r"scored\s+(\d\.?\d?)\s*(out of)?\s*5",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            val = float(match.group(1))
            if 1 <= val <= 5:
                return val
    return 3.0


def parse_resume(uploaded_file):
    text = extract_text_from_pdf(uploaded_file)

    extracted = {
        "department": extract_department(text),
        "education": extract_education(text),
        "gender": extract_gender(text),
        "no_of_trainings": extract_trainings(text),
        "age": extract_age(text) or 30,
        "previous_year_rating": extract_previous_rating(text),
        "length_of_service": extract_experience(text) or 3,
        "awards_won": extract_awards(text),
        "avg_training_score": extract_avg_training_score(text),
    }

    confidence = {}
    confidence["department"] = "high" if extract_department(text) != "Technology" else "low"
    confidence["age"] = "high" if extract_age(text) else "estimated"
    confidence["length_of_service"] = "high" if extract_experience(text) else "estimated"
    confidence["education"] = "high"
    confidence["avg_training_score"] = "estimated"

    return extracted, confidence, text[:2000]
