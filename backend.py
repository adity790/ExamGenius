"""
Backend Module: Core Business Logic for Indian Exam Recommendation System
Version: 3.0 - Streamlined version without AI, focused on stream and class-based filtering
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date
from dataclasses import dataclass, asdict
from enum import Enum
import json
import hashlib

import joblib
import os
import numpy as np


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== DATA MODELS ==========

class Stream(Enum):
    """Academic stream enumeration"""
    SCIENCE_PCM = "Science (PCM)"
    SCIENCE_PCB = "Science (PCB)"
    SCIENCE_PCMB = "Science (PCMB)"
    COMMERCE = "Commerce"
    ARTS = "Arts/Humanities"
    LAW = "Law"
    DESIGN = "Design"
    ARCHITECTURE = "Architecture"
    MANAGEMENT = "Management"
    HOTEL_MGMT = "Hotel Management"
    EDUCATION = "Education"
    PHARMACY = "Pharmacy"
    NURSING = "Nursing"
    AGRICULTURE = "Agriculture"
    VETERINARY = "Veterinary"
    COMPUTER_SCIENCE = "IT / Computer Science"
    DEFENCE = "Defence"
    RESEARCH = "Research"
    VOCATIONAL = "Vocational / Skill-based"
    OTHER = "Other"
    ALL = "All"

@dataclass
class StudentProfile:
    """Student profile data model"""
    current_class: str
    stream: str
    state: str
    board: str
    scholarship_needed: bool
    age: Optional[int] = None
    percentage: Optional[float] = None

    def __post_init__(self):
        """Validate student profile after initialization"""
        if self.percentage and (self.percentage < 0 or self.percentage > 100):
            raise ValueError("Percentage must be between 0 and 100")

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)

@dataclass
class Exam:
    """Exam data model with validation"""
    name: str
    conducting_body: str
    stream: str
    level: str
    eligible_from: str
    min_percentage: Optional[float] = None
    age_limit_min: Optional[float] = None
    age_limit_max: Optional[float] = None
    exam_mode: str = "Offline"
    scholarship_available: bool = False
    last_date: Optional[date] = None
    application_fee: Optional[float] = None
    syllabus_link: Optional[str] = None
    frequency: str = "Annual"
    exam_category: str = "General"
    state_specific: Optional[str] = None

    def is_eligible(self, student: StudentProfile) -> bool:
        """Check if student is eligible for this exam"""
        # Stream check - if exam stream is "All", it matches any student stream
        if self.stream != "All" and self.stream != student.stream:
            return False

        # Class eligibility check using the provided mapping
        eligibility_map = {
            '8': ['8', '9', '10', '11', '12', 'Diploma', 'UG', 'PG', 'PhD'],
            '9': ['9', '10', '11', '12', 'Diploma', 'UG', 'PG', 'PhD'],
            '10': ['10', '11', '12', 'Diploma', 'UG', 'PG', 'PhD'],
            '11': ['11', '12', 'Diploma', 'UG', 'PG', 'PhD'],
            '12': ['12', 'Diploma', 'UG', 'PG', 'PhD'],
            'Diploma': ['Diploma', 'UG', 'PG', 'PhD'],
            'UG': ['UG', 'PG', 'PhD'],
            'PG': ['PG', 'PhD'],
            'PhD': ['PhD']
        }
        
        # Check if student's current class is eligible for this exam
        if student.current_class in eligibility_map:
            eligible_classes = eligibility_map[student.current_class]
            if self.eligible_from not in eligible_classes:
                return False
        else:
            # If class not in map, use direct comparison
            class_order = ["8", "9", "10", "11", "12", "Diploma", "UG", "PG", "PhD"]
            try:
                student_idx = class_order.index(student.current_class)
                eligible_idx = class_order.index(self.eligible_from)
                if student_idx < eligible_idx:
                    return False
            except ValueError:
                return False

        # Percentage check
        if self.min_percentage is not None and student.percentage is not None:
            if student.percentage < self.min_percentage:
                return False

        # Age check
        if student.age is not None and self.age_limit_min is not None and self.age_limit_max is not None:
            if not (self.age_limit_min <= student.age <= self.age_limit_max):
                return False

        # Scholarship check
        if student.scholarship_needed and not self.scholarship_available:
            return False

        return True

# ========== GOVERNMENT EXAMS DATABASE ==========

class GovernmentExamDatabase:
    """Comprehensive government exam database with 100+ exams"""

    # State-Level Exam Conducting Bodies
    STATE_BODIES = {
        "MAHARASHTRA": "MPSC",
        "UTTAR_PRADESH": "UPPSC",
        "MADHYA_PRADESH": "MPPSC",
        "RAJASTHAN": "RPSC",
        "TAMIL_NADU": "TNPSC",
        "KARNATAKA": "KPSC",
        "KERALA": "Kerala PSC",
        "WEST_BENGAL": "WBPSC",
        "GUJARAT": "GPSC",
        "BIHAR": "BPSC",
        "ANDHRA_PRADESH": "APPSC",
        "TELANGANA": "TSPSC",
        "ODISHA": "OPSC",
        "PUNJAB": "PPSC",
        "HARYANA": "HPSC"
    }

    def __init__(self):
        self._exams = self._load_all_exams()
        self._exam_cache = {}

        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "ml", "exam_ranker.pkl")
            self.ml_model = joblib.load(model_path)
            logger.info("✅ ML ranking model loaded successfully")
        except Exception as e:
            self.ml_model = None
            logger.warning(f"⚠️ ML model not loaded, falling back to rule-based logic: {e}")

    def _load_all_exams(self) -> List[Exam]:
        """Load all government exams from the provided list"""
        exams = []
        today = date.today()
        
        # ===== SCHOOL LEVEL EXAMS (Class 5-12) =====
        school_exams = [
            # Kishore Vaigyanik Protsahan Yojana (KVPY)
            Exam(
                name="Kishore Vaigyanik Protsahan Yojana (KVPY)",
                conducting_body="Department of Science & Technology (DST) / IISc",
                stream=Stream.SCIENCE_PCMB.value,
                level="School",
                eligible_from="11",
                scholarship_available=True,
                frequency="Annual",
                last_date=today.replace(month=8, day=31),
                exam_category="Scholarship & Research"
            ),
            # National Talent Search Examination (NTSE)
            Exam(
                name="National Talent Search Examination (NTSE)",
                conducting_body="NCERT",
                stream="All",
                level="School",
                eligible_from="10",
                min_percentage=60.0,
                scholarship_available=True,
                frequency="Annual",
                last_date=today.replace(month=10, day=31),
                exam_category="Scholarship"
            ),
            # National Means-cum-Merit Scholarship (NMMS)
            Exam(
                name="National Means-cum-Merit Scholarship (NMMS)",
                conducting_body="Ministry of Education / SCERT",
                stream="All",
                level="School",
                eligible_from="8",
                scholarship_available=True,
                frequency="Annual",
                last_date=today.replace(month=11, day=30),
                exam_category="Scholarship"
            ),
            # National Science Olympiad (NSO)
            Exam(
                name="National Science Olympiad (NSO)",
                conducting_body="Science Olympiad Foundation",
                stream=Stream.SCIENCE_PCMB.value,
                level="School",
                eligible_from="8",
                frequency="Annual",
                last_date=today.replace(month=9, day=15),
                exam_category="Olympiad"
            ),
            # Indian National Olympiad (INO)
            Exam(
                name="Indian National Olympiad (INO)",
                conducting_body="HBCSE / IAPT",
                stream=Stream.SCIENCE_PCMB.value,
                level="School",
                eligible_from="8",
                frequency="Annual",
                last_date=today.replace(month=8, day=31),
                exam_category="Olympiad"
            ),
            # Junior Science Talent Search Examination (JSTSE)
            Exam(
                name="Junior Science Talent Search Examination (JSTSE)",
                conducting_body="State Governments",
                stream=Stream.SCIENCE_PCMB.value,
                level="School",
                eligible_from="9",
                frequency="Annual",
                last_date=today.replace(month=10, day=15),
                exam_category="Scholarship"
            ),
            # Mukhyamantri Vigyan Pratibha Pariksha
            Exam(
                name="Mukhyamantri Vigyan Pratibha Pariksha",
                conducting_body="Delhi Directorate of Education",
                stream=Stream.SCIENCE_PCMB.value,
                level="School",
                eligible_from="9",
                scholarship_available=True,
                frequency="Annual",
                last_date=today.replace(month=9, day=30),
                exam_category="Scholarship",
                state_specific="Delhi"
            ),
            # CBSE Board Examination
            Exam(
                name="CBSE Board Examination",
                conducting_body="CBSE",
                stream="All",
                level="School",
                eligible_from="10",
                frequency="Annual",
                last_date=today.replace(month=12, day=31),
                exam_category="Board Exam"
            )
        ]
        exams.extend(school_exams)

        # ===== UNDERGRADUATE ENTRANCE EXAMS =====
        ug_exams = [
            # JEE Main
            Exam(
                name="JEE Main",
                conducting_body="National Testing Agency (NTA)",
                stream=Stream.SCIENCE_PCM.value,
                level="UG",
                eligible_from="12",
                age_limit_min=17.0,
                age_limit_max=25.0,
                exam_mode="Computer Based",
                frequency="Twice Yearly",
                last_date=today.replace(month=12, day=15),
                exam_category="Engineering"
            ),
            # JEE Advanced
            Exam(
                name="JEE Advanced",
                conducting_body="IITs (JAB)",
                stream=Stream.SCIENCE_PCM.value,
                level="UG",
                eligible_from="12",
                frequency="Annual",
                last_date=today.replace(month=4, day=30),
                exam_category="Engineering"
            ),
            # NEET-UG
            Exam(
                name="NEET-UG",
                conducting_body="NTA",
                stream=Stream.SCIENCE_PCB.value,
                level="UG",
                eligible_from="12",
                age_limit_min=17.0,
                age_limit_max=25.0,
                exam_mode="Pen & Paper",
                frequency="Annual",
                last_date=today.replace(month=3, day=31),
                exam_category="Medical"
            ),
            # CUET-UG
            Exam(
                name="CUET-UG",
                conducting_body="NTA",
                stream="All",
                level="UG",
                eligible_from="12",
                frequency="Annual",
                last_date=today.replace(month=4, day=30),
                exam_category="University Entrance"
            ),
            # NCHM JEE
            Exam(
                name="NCHM JEE",
                conducting_body="NTA",
                stream=Stream.HOTEL_MGMT.value,
                level="UG",
                eligible_from="12",
                frequency="Annual",
                last_date=today.replace(month=4, day=30),
                exam_category="Hotel Management"
            ),
            # WBJEE
            Exam(
                name="WBJEE",
                conducting_body="WBJEEB",
                stream=Stream.SCIENCE_PCM.value,
                level="UG",
                eligible_from="12",
                frequency="Annual",
                last_date=today.replace(month=1, day=31),
                exam_category="Engineering",
                state_specific="West Bengal"
            ),
            # MHT-CET
            Exam(
                name="MHT-CET",
                conducting_body="Govt of Maharashtra (DTE)",
                stream=Stream.SCIENCE_PCM.value,
                level="UG",
                eligible_from="12",
                frequency="Annual",
                last_date=today.replace(month=3, day=31),
                exam_category="Engineering",
                state_specific="Maharashtra"
            ),
            # KEAM
            Exam(
                name="KEAM",
                conducting_body="CEE Kerala",
                stream=Stream.SCIENCE_PCM.value,
                level="UG",
                eligible_from="12",
                frequency="Annual",
                last_date=today.replace(month=3, day=31),
                exam_category="Engineering",
                state_specific="Kerala"
            ),
            # COMEDK UGET
            Exam(
                name="COMEDK UGET",
                conducting_body="COMEDK",
                stream=Stream.SCIENCE_PCM.value,
                level="UG",
                eligible_from="12",
                frequency="Annual",
                last_date=today.replace(month=4, day=30),
                exam_category="Engineering"
            ),
            # IISER Aptitude Test
            Exam(
                name="IISER Aptitude Test",
                conducting_body="IISERs",
                stream=Stream.SCIENCE_PCMB.value,
                level="UG",
                eligible_from="12",
                frequency="Annual",
                last_date=today.replace(month=4, day=30),
                exam_category="Research"
            )
        ]
        exams.extend(ug_exams)

        # ===== POSTGRADUATE ENTRANCE EXAMS =====
        pg_exams = [
            # GATE
            Exam(
                name="GATE",
                conducting_body="IITs",
                stream=Stream.SCIENCE_PCM.value,
                level="PG",
                eligible_from="UG",
                frequency="Annual",
                last_date=today.replace(month=10, day=31),
                exam_category="Engineering PG"
            ),
            # CAT
            Exam(
                name="CAT",
                conducting_body="IIMs",
                stream=Stream.MANAGEMENT.value,
                level="PG",
                eligible_from="UG",
                frequency="Annual",
                last_date=today.replace(month=9, day=30),
                exam_category="Management"
            ),
            # MAT
            Exam(
                name="MAT",
                conducting_body="AIMA",
                stream=Stream.MANAGEMENT.value,
                level="PG",
                eligible_from="UG",
                frequency="Multiple times a year",
                last_date=today.replace(month=12, day=31),
                exam_category="Management"
            ),
            # XAT
            Exam(
                name="XAT",
                conducting_body="XLRI",
                stream=Stream.MANAGEMENT.value,
                level="PG",
                eligible_from="UG",
                frequency="Annual",
                last_date=today.replace(month=11, day=30),
                exam_category="Management"
            ),
            # CMAT
            Exam(
                name="CMAT",
                conducting_body="NTA",
                stream=Stream.MANAGEMENT.value,
                level="PG",
                eligible_from="UG",
                frequency="Annual",
                last_date=today.replace(month=3, day=31),
                exam_category="Management"
            ),
            # GPAT
            Exam(
                name="GPAT",
                conducting_body="NTA",
                stream=Stream.PHARMACY.value,
                level="PG",
                eligible_from="UG",
                frequency="Annual",
                last_date=today.replace(month=11, day=30),
                exam_category="Pharmacy"
            ),
            # NIMCET
            Exam(
                name="NIMCET",
                conducting_body="NITs",
                stream=Stream.COMPUTER_SCIENCE.value,
                level="PG",
                eligible_from="UG",
                frequency="Annual",
                last_date=today.replace(month=4, day=30),
                exam_category="Computer Science"
            ),
            # NEET-PG
            Exam(
                name="NEET-PG",
                conducting_body="NBE",
                stream="Medical",
                level="PG",
                eligible_from="UG",
                frequency="Annual",
                last_date=today.replace(month=11, day=15),
                exam_category="Medical"
            ),
            # UGC NET
            Exam(
                name="UGC NET",
                conducting_body="NTA / UGC",
                stream=Stream.RESEARCH.value,
                level="PG",
                eligible_from="PG",
                min_percentage=55.0,
                frequency="Twice Yearly",
                last_date=today.replace(month=4, day=30),
                exam_category="Teaching & Research"
            )
        ]
        exams.extend(pg_exams)

        # ===== DEFENCE & ARMED FORCES EXAMS =====
        defence_exams = [
            # NDA
            Exam(
                name="NDA",
                conducting_body="UPSC",
                stream=Stream.DEFENCE.value,
                level="UG",
                eligible_from="12",
                age_limit_min=16.5,
                age_limit_max=19.5,
                exam_mode="Pen & Paper",
                frequency="Twice Yearly",
                last_date=today.replace(month=6, day=30),
                exam_category="Defence"
            ),
            # CDS
            Exam(
                name="CDS",
                conducting_body="UPSC",
                stream=Stream.DEFENCE.value,
                level="UG",
                eligible_from="UG",
                exam_mode="Pen & Paper",
                frequency="Twice Yearly",
                last_date=today.replace(month=10, day=31),
                exam_category="Defence"
            ),
            # AFCAT
            Exam(
                name="AFCAT",
                conducting_body="Indian Air Force",
                stream=Stream.DEFENCE.value,
                level="UG",
                eligible_from="UG",
                frequency="Twice Yearly",
                last_date=today.replace(month=12, day=31),
                exam_category="Defence"
            ),
            # TES
            Exam(
                name="TES",
                conducting_body="Indian Army",
                stream=Stream.SCIENCE_PCM.value,
                level="UG",
                eligible_from="12",
                frequency="Twice Yearly",
                last_date=today.replace(month=6, day=30),
                exam_category="Defence"
            )
        ]
        exams.extend(defence_exams)

        # ===== CENTRAL GOVERNMENT JOB EXAMS =====
        govt_job_exams = [
            # UPSC Civil Services (CSE)
            Exam(
                name="UPSC Civil Services (CSE)",
                conducting_body="UPSC",
                stream="All",
                level="PG",
                eligible_from="UG",
                age_limit_min=21.0,
                age_limit_max=32.0,
                exam_mode="Pen & Paper",
                frequency="Annual",
                last_date=today.replace(month=3, day=31),
                exam_category="Government Job"
            ),
            # UPSC CAPF AC
            Exam(
                name="UPSC CAPF AC",
                conducting_body="UPSC",
                stream="All",
                level="UG",
                eligible_from="UG",
                frequency="Annual",
                last_date=today.replace(month=4, day=30),
                exam_category="Government Job"
            ),
            # SSC CGL
            Exam(
                name="SSC CGL",
                conducting_body="SSC",
                stream="All",
                level="UG",
                eligible_from="UG",
                age_limit_min=18.0,
                age_limit_max=32.0,
                exam_mode="Computer Based",
                frequency="Annual",
                last_date=today.replace(month=10, day=31),
                exam_category="Government Job"
            ),
            # SSC CHSL
            Exam(
                name="SSC CHSL",
                conducting_body="SSC",
                stream="All",
                level="UG",
                eligible_from="12",
                age_limit_min=18.0,
                age_limit_max=27.0,
                exam_mode="Computer Based",
                frequency="Annual",
                last_date=today.replace(month=11, day=30),
                exam_category="Government Job"
            ),
            # SSC JE
            Exam(
                name="SSC JE",
                conducting_body="SSC",
                stream=Stream.SCIENCE_PCM.value,
                level="UG",
                eligible_from="Diploma",
                frequency="Annual",
                last_date=today.replace(month=10, day=31),
                exam_category="Government Job"
            ),
            # SSC GD Constable
            Exam(
                name="SSC GD Constable",
                conducting_body="SSC",
                stream="All",
                level="UG",
                eligible_from="10",
                frequency="Annual",
                last_date=today.replace(month=12, day=31),
                exam_category="Government Job"
            ),
            # SSC Stenographer
            Exam(
                name="SSC Stenographer",
                conducting_body="SSC",
                stream="All",
                level="UG",
                eligible_from="12",
                frequency="Annual",
                last_date=today.replace(month=11, day=30),
                exam_category="Government Job"
            ),
            # RRB NTPC
            Exam(
                name="RRB NTPC",
                conducting_body="Railway Recruitment Boards",
                stream="All",
                level="UG",
                eligible_from="12",
                frequency="Annual",
                last_date=today.replace(month=4, day=30),
                exam_category="Government Job"
            ),
            # RRB JE / SE
            Exam(
                name="RRB JE / SE",
                conducting_body="RRB",
                stream=Stream.SCIENCE_PCM.value,
                level="UG",
                eligible_from="Diploma",
                frequency="Annual",
                last_date=today.replace(month=3, day=31),
                exam_category="Government Job"
            )
        ]
        exams.extend(govt_job_exams)

        # ===== BANKING & FINANCIAL SECTOR EXAMS =====
        banking_exams = [
            # IBPS PO
            Exam(
                name="IBPS PO",
                conducting_body="IBPS",
                stream=Stream.MANAGEMENT.value,
                level="UG",
                eligible_from="UG",
                age_limit_min=20.0,
                age_limit_max=30.0,
                exam_mode="Computer Based",
                frequency="Annual",
                last_date=today.replace(month=8, day=31),
                exam_category="Banking"
            ),
            # IBPS Clerk
            Exam(
                name="IBPS Clerk",
                conducting_body="IBPS",
                stream="All",
                level="UG",
                eligible_from="UG",
                frequency="Annual",
                last_date=today.replace(month=9, day=30),
                exam_category="Banking"
            ),
            # SBI PO
            Exam(
                name="SBI PO",
                conducting_body="State Bank of India",
                stream=Stream.MANAGEMENT.value,
                level="UG",
                eligible_from="UG",
                frequency="Annual",
                last_date=today.replace(month=10, day=31),
                exam_category="Banking"
            ),
            # SBI Clerk
            Exam(
                name="SBI Clerk",
                conducting_body="State Bank of India",
                stream="All",
                level="UG",
                eligible_from="UG",
                frequency="Annual",
                last_date=today.replace(month=11, day=30),
                exam_category="Banking"
            ),
            # RBI Grade B
            Exam(
                name="RBI Grade B",
                conducting_body="Reserve Bank of India",
                stream=Stream.MANAGEMENT.value,
                level="UG",
                eligible_from="UG",
                frequency="Annual",
                last_date=today.replace(month=5, day=31),
                exam_category="Banking"
            ),
            # NABARD Grade A
            Exam(
                name="NABARD Grade A",
                conducting_body="NABARD",
                stream=Stream.AGRICULTURE.value,
                level="UG",
                eligible_from="UG",
                frequency="Annual",
                last_date=today.replace(month=6, day=30),
                exam_category="Banking"
            )
        ]
        exams.extend(banking_exams)

        # ===== TEACHING & EDUCATION EXAMS =====
        teaching_exams = [
            # CTET
            Exam(
                name="CTET",
                conducting_body="CBSE",
                stream=Stream.EDUCATION.value,
                level="PG",
                eligible_from="UG",
                frequency="Twice Yearly",
                last_date=today.replace(month=9, day=30),
                exam_category="Teaching"
            ),
            # KVS TGT / PGT
            Exam(
                name="KVS TGT / PGT",
                conducting_body="Kendriya Vidyalaya Sangathan",
                stream=Stream.EDUCATION.value,
                level="PG",
                eligible_from="UG",
                frequency="Annual",
                last_date=today.replace(month=5, day=31),
                exam_category="Teaching"
            )
        ]
        exams.extend(teaching_exams)

        # ===== LAW ENTRANCE EXAMS =====
        law_exams = [
            # CLAT UG
            Exam(
                name="CLAT UG",
                conducting_body="CLAT Consortium",
                stream=Stream.LAW.value,
                level="UG",
                eligible_from="12",
                age_limit_min=18.0,
                exam_mode="Computer Based",
                frequency="Annual",
                last_date=today.replace(month=11, day=30),
                exam_category="Law"
            ),
            # AILET
            Exam(
                name="AILET",
                conducting_body="NLU Delhi",
                stream=Stream.LAW.value,
                level="UG",
                eligible_from="12",
                frequency="Annual",
                last_date=today.replace(month=12, day=31),
                exam_category="Law"
            )
        ]
        exams.extend(law_exams)

        # ===== STATE PUBLIC SERVICE COMMISSION EXAMS =====
        state_exams = [
            # UPPSC
            Exam(
                name="UPPSC",
                conducting_body="Uttar Pradesh PSC",
                stream="All",
                level="PG",
                eligible_from="UG",
                frequency="Annual",
                last_date=today.replace(month=4, day=30),
                exam_category="State Government Job",
                state_specific="Uttar Pradesh"
            ),
            # MPSC
            Exam(
                name="MPSC",
                conducting_body="Maharashtra PSC",
                stream="All",
                level="PG",
                eligible_from="UG",
                frequency="Annual",
                last_date=today.replace(month=5, day=31),
                exam_category="State Government Job",
                state_specific="Maharashtra"
            ),
            # TNPSC
            Exam(
                name="TNPSC",
                conducting_body="Tamil Nadu PSC",
                stream="All",
                level="PG",
                eligible_from="UG",
                frequency="Annual",
                last_date=today.replace(month=6, day=30),
                exam_category="State Government Job",
                state_specific="Tamil Nadu"
            ),
            # BPSC
            Exam(
                name="BPSC",
                conducting_body="Bihar PSC",
                stream="All",
                level="PG",
                eligible_from="UG",
                frequency="Annual",
                last_date=today.replace(month=5, day=31),
                exam_category="State Government Job",
                state_specific="Bihar"
            ),
            # RPSC
            Exam(
                name="RPSC",
                conducting_body="Rajasthan PSC",
                stream="All",
                level="PG",
                eligible_from="UG",
                frequency="Annual",
                last_date=today.replace(month=6, day=30),
                exam_category="State Government Job",
                state_specific="Rajasthan"
            ),
            # KPSC
            Exam(
                name="KPSC",
                conducting_body="Karnataka PSC",
                stream="All",
                level="PG",
                eligible_from="UG",
                frequency="Annual",
                last_date=today.replace(month=5, day=31),
                exam_category="State Government Job",
                state_specific="Karnataka"
            )
        ]
        exams.extend(state_exams)

        # ===== PROFESSIONAL & STATUTORY EXAMS =====
        professional_exams = [
            # CA Foundation
            Exam(
                name="CA Foundation",
                conducting_body="ICAI",
                stream=Stream.COMMERCE.value,
                level="Professional",
                eligible_from="12",
                frequency="Twice Yearly",
                last_date=today.replace(month=7, day=1),
                exam_category="Professional"
            ),
            # CS Executive
            Exam(
                name="CS Executive",
                conducting_body="ICSI",
                stream=Stream.COMMERCE.value,
                level="Professional",
                eligible_from="12",
                frequency="Twice Yearly",
                last_date=today.replace(month=8, day=31),
                exam_category="Professional"
            ),
            # CMA Foundation
            Exam(
                name="CMA Foundation",
                conducting_body="ICMAI",
                stream=Stream.COMMERCE.value,
                level="Professional",
                eligible_from="12",
                frequency="Twice Yearly",
                last_date=today.replace(month=6, day=30),
                exam_category="Professional"
            )
        ]
        exams.extend(professional_exams)

        return exams
    def _build_ml_features(self, student: StudentProfile, exam: Exam) -> List[float]:
        """
            Convert student + exam into ML feature vector
            """
        return [
            1 if exam.stream == student.stream else 0,
            1 if exam.stream == "All" else 0,
            student.percentage or 0,
            exam.min_percentage or 0,
            1 if exam.scholarship_available else 0,
            1 if student.scholarship_needed else 0,
            1 if exam.state_specific and student.state.lower() == exam.state_specific.lower() else 0,
            exam.application_fee or 0
        ]


    def get_exams_for_student(self, student: StudentProfile) -> List[Exam]:
        """Get all government exams eligible for a student based on stream and class"""
        cache_key = self._generate_cache_key(student)

        if cache_key in self._exam_cache:
            return self._exam_cache[cache_key]

        eligible_exams = []
        for exam in self._exams:
            try:
                if exam.is_eligible(student):
                    # ML-based scoring
                    if self.ml_model:
                        features = self._build_ml_features(student, exam)
                        score = float(self.ml_model.predict_proba([features])[0][1])
                    else:
                        # fallback score
                        score = 0.5

                    eligible_exams.append((score, exam))

            except Exception as e:
                logger.error(f"Error checking eligibility for {exam.name}: {e}")

        eligible_exams.sort(key=lambda x: x[0], reverse=True)

        ranked_exams = []

        for idx, (score, exam) in enumerate(eligible_exams, start=1):
            exam.ml_rank = idx                 # ✅ ADD RANK
            exam.ml_enhanced_score = score     # ✅ STORE SCORE
            ranked_exams.append(exam)

        # Cache for performance
        self._exam_cache[cache_key] = ranked_exams

        return ranked_exams
    
    def get_exams_by_stream(self, stream: str) -> List[Exam]:
        """Get exams by academic stream"""
        return [exam for exam in self._exams if exam.stream == stream or exam.stream == "All"]

    def get_exams_by_level(self, level: str) -> List[Exam]:
        """Get exams by education level"""
        return [exam for exam in self._exams if exam.level == level]

    def get_exams_by_category(self, category: str) -> List[Exam]:
        """Get exams by category"""
        return [exam for exam in self._exams if exam.exam_category == category]

    def _generate_cache_key(self, student: StudentProfile) -> str:
        """Generate cache key from student profile"""
        profile_str = (
            f"{student.current_class}_{student.stream}_"
            f"{student.state}_{student.board}_{student.scholarship_needed}_"
            f"{student.age or 'None'}_{student.percentage or 'None'}"
        )
        return hashlib.md5(profile_str.encode()).hexdigest()

    def get_total_exams_count(self) -> int:
        """Get total number of exams in database"""
        return len(self._exams)

    def get_exams_statistics(self) -> Dict:
        """Get statistics about exams"""
        stats = {
            "total_exams": len(self._exams),
            "by_level": {},
            "by_category": {},
            "by_stream": {},
            "scholarship_exams": 0
        }
        
        for exam in self._exams:
            # By level
            stats["by_level"][exam.level] = stats["by_level"].get(exam.level, 0) + 1
            
            # By category
            stats["by_category"][exam.exam_category] = stats["by_category"].get(exam.exam_category, 0) + 1
            
            # By stream
            stats["by_stream"][exam.stream] = stats["by_stream"].get(exam.stream, 0) + 1
            
            # Scholarship exams
            if exam.scholarship_available:
                stats["scholarship_exams"] += 1
        
        return stats

# ========== EXAM ANALYTICS ENGINE ==========

class ExamAnalytics:
    """Analytics for exam recommendations"""

    def __init__(self):
        self.recommendation_history = []

    def calculate_success_probability(self, student: StudentProfile, exam: Exam) -> float:
        """Calculate probability of success in exam"""
        probability = 0.5  # Base probability

        # Stream match
        if exam.stream == student.stream:
            probability += 0.2
        elif exam.stream == "All":
            probability += 0.1

        # Academic performance
        if student.percentage is not None:
            if student.percentage >= 90:
                probability += 0.2
            elif student.percentage >= 80:
                probability += 0.15
            elif student.percentage >= 70:
                probability += 0.1
            elif student.percentage >= 60:
                probability += 0.05

        # Age appropriateness
        if student.age is not None and exam.age_limit_min is not None and exam.age_limit_max is not None:
            if exam.age_limit_min <= student.age <= exam.age_limit_max:
                probability += 0.1

        return max(0.05, min(probability, 0.95))

    def generate_comparison_matrix(self, exams: List[Exam]) -> pd.DataFrame:
        """Generate comparison matrix for exams"""
        data = []
        for exam in exams:
            data.append({
                "Exam": exam.name,
                "Conducting Body": exam.conducting_body,
                "Stream": exam.stream,
                "Level": exam.level,
                "Eligibility": f"{exam.eligible_from}+",
                "Scholarship": "Yes" if exam.scholarship_available else "No",
                "Mode": exam.exam_mode,
                "Last Date": exam.last_date.strftime('%Y-%m-%d') if exam.last_date else "N/A",
                "Category": exam.exam_category
            })

        return pd.DataFrame(data)

    def track_recommendation(self, student_hash: str, exam_name: str, action: str):
        """Track recommendation interactions"""
        self.recommendation_history.append({
            "timestamp": datetime.now(),
            "student_hash": student_hash,
            "exam": exam_name,
            "action": action,
            "platform": "streamlit_app"
        })

# ========== MAIN BACKEND CLASS ==========

class ExamRecommendationSystem:
    """Main backend orchestrator for the recommendation system"""

    def __init__(self):
        self.govt_db = GovernmentExamDatabase()
        self.analytics = ExamAnalytics()
        logger.info(f"Exam Recommendation System initialized with {self.govt_db.get_total_exams_count()} exams")

    def process_student_query(
        self,
        current_class: str,
        stream: str,
        state: str,
        board: str,
        scholarship_needed: bool,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main method to process student query and return recommendations
        """
        try:
            # Create student profile
            student = StudentProfile(
                current_class=current_class,
                stream=stream,
                state=state,
                board=board,
                scholarship_needed=scholarship_needed,
                age=kwargs.get('age'),
                percentage=kwargs.get('percentage')
            )

            # Get eligible government exams
            govt_exams = self.govt_db.get_exams_for_student(student)

            # Get university exams if provided
            university_exams = kwargs.get('university_exams', [])

            # Generate comparison matrix
            exams_for_comparison = govt_exams[:10] if len(govt_exams) >= 10 else govt_exams
            comparison_df = self.analytics.generate_comparison_matrix(exams_for_comparison)

            # Calculate success probabilities
            success_probs = []
            exams_for_prob = govt_exams[:5] if len(govt_exams) >= 5 else govt_exams
            for exam in exams_for_prob:
                prob = self.analytics.calculate_success_probability(student, exam)
                success_probs.append({
                    "exam": exam.name,
                    "success_probability": f"{prob:.1%}",
                    "key_factors": self._get_success_factors(student, exam)
                })

            # Prepare response
            response = {
                "student_profile": student.to_dict(),
                "eligible_exams_count": len(govt_exams),
                "government_exams": [
                    {
                        "name": exam.name,
                        "conducting_body": exam.conducting_body,
                        "level": exam.level,
                        "eligible_from": exam.eligible_from,
                        "scholarship": exam.scholarship_available,
                        "stream": exam.stream,
                        "min_percentage": exam.min_percentage,
                        "age_limit_min": exam.age_limit_min,
                        "age_limit_max": exam.age_limit_max,
                        "exam_mode": exam.exam_mode,
                        "last_date": exam.last_date.isoformat() if exam.last_date else None,
                        "application_fee": exam.application_fee,
                        "frequency": exam.frequency,
                        "exam_category": exam.exam_category,
                        "state_specific": exam.state_specific
                    }
                    for exam in govt_exams  
                ],
                "university_exams": university_exams,
                "analytics": {
                    "comparison_matrix": comparison_df.to_dict('records'),
                    "success_probabilities": success_probs,
                    "top_recommendations": self._get_top_recommendations(govt_exams, student),
                    "exam_statistics": self.govt_db.get_exams_statistics()
                },
                "metadata": {
                    "processed_at": datetime.now().isoformat(),
                    "api_version": "3.0",
                    "total_exams_in_db": self.govt_db.get_total_exams_count()
                }
            }

            logger.info(f"Processed query: Class {current_class}, Stream {stream}, Found {len(govt_exams)} exams")
            return response

        except Exception as e:
            logger.error(f"Error processing student query: {e}")
            return self._get_error_response(e)

    def _get_success_factors(self, student: StudentProfile, exam: Exam) -> List[str]:
        """Identify key success factors"""
        factors = []

        if exam.stream == student.stream:
            factors.append("Perfect stream match")
        elif exam.stream == "All":
            factors.append("Open to all streams")

        if student.percentage is not None and exam.min_percentage is not None:
            if student.percentage >= exam.min_percentage + 10:
                factors.append("Excellent academic record")
            elif student.percentage >= exam.min_percentage:
                factors.append("Meets minimum criteria")

        if student.scholarship_needed and exam.scholarship_available:
            factors.append("Scholarship available")

        if student.age is not None and exam.age_limit_min is not None and exam.age_limit_max is not None:
            if exam.age_limit_min <= student.age <= exam.age_limit_max:
                factors.append("Within ideal age range")

        return factors or ["Standard preparation needed"]

    def _get_top_recommendations(self, exams: List[Exam], student: StudentProfile) -> List[Dict]:
        """Get top recommendations with reasoning"""
        if not exams:
            return []

        top_exams = exams[:5]
        recommendations = []

        for i, exam in enumerate(top_exams, 1):
            rec = {
                "rank": i,
                "exam": exam.name,
                "why_recommended": [],
                "action_items": []
            }

            # Add reasoning
            if exam.stream == student.stream:
                rec["why_recommended"].append("Perfect match with your academic stream")
            elif exam.stream == "All":
                rec["why_recommended"].append("Open to all academic backgrounds")

            # Scholarship match
            if student.scholarship_needed and exam.scholarship_available:
                rec["why_recommended"].append("Includes scholarship/financial aid")

            # State-specific advantage
            if student.state != "All-India" and exam.state_specific:
                if student.state.lower() == exam.state_specific.lower():
                    rec["why_recommended"].append("State-specific exam - higher chances")

            # Urgency based on deadline
            if exam.last_date:
                days_remaining = (exam.last_date - date.today()).days
                if days_remaining <= 30:
                    rec["why_recommended"].append(f"Application closes in {days_remaining} days")
                    rec["action_items"].append(f"Apply immediately before {exam.last_date.strftime('%d %b %Y')}")
                elif days_remaining <= 60:
                    rec["why_recommended"].append(f"Application closes in {days_remaining} days")
                    rec["action_items"].append(f"Apply by: {exam.last_date.strftime('%d %b %Y')}")

            # Additional action items
            rec["action_items"].append(f"Check {exam.conducting_body} website for details")
            rec["action_items"].append("Prepare according to exam pattern and syllabus")

            recommendations.append(rec)

        return recommendations

    def _get_error_response(self, error: Exception) -> Dict:
        """Generate error response"""
        return {
            "error": True,
            "message": str(error),
            "fallback_recommendations": {
                "message": "Unable to process request. Please check your inputs.",
                "general_exams": [
                    "Check NTA website for national-level exams",
                    "Visit state education board website for state-level exams",
                    "Explore National Scholarship Portal for financial aid"
                ]
            }
        }

    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        stats = self.govt_db.get_exams_statistics()
        return {
            "total_government_exams": stats["total_exams"],
            "exam_levels": stats["by_level"],
            "exam_categories": stats["by_category"],
            "streams_covered": stats["by_stream"],
            "scholarship_exams": stats["scholarship_exams"],
            "state_bodies_covered": len(self.govt_db.STATE_BODIES),
            "cache_size": len(self.govt_db._exam_cache)
        }

# ========== UTILITY FUNCTIONS ==========

def validate_student_data(data: Dict) -> Tuple[bool, List[str]]:
    """Validate student input data"""
    errors = []

    required_fields = ['current_class', 'stream', 'state', 'board']
    for field in required_fields:
        if field not in data or not data[field]:
            errors.append(f"Missing required field: {field}")

    if 'current_class' in data:
        valid_classes = ["8", "9", "10", "11", "12", "Diploma", "UG", "PG", "PhD"]
        if data['current_class'] not in valid_classes:
            errors.append(f"Invalid class: {data['current_class']}. Must be one of {valid_classes}")

    if 'percentage' in data and data['percentage'] is not None:
        try:
            percentage = float(data['percentage'])
            if percentage < 0 or percentage > 100:
                errors.append("Percentage must be between 0 and 100")
        except (ValueError, TypeError):
            errors.append("Percentage must be a number")

    if 'age' in data and data['age'] is not None:
        try:
            age = int(data['age'])
            if age < 10 or age > 50:
                errors.append("Age must be between 10 and 50 for educational purposes")
        except (ValueError, TypeError):
            errors.append("Age must be an integer")

    return len(errors) == 0, errors

def export_recommendations_to_json(recommendations: Dict, filename: str = None):
    """Export recommendations to JSON file"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exam_recommendations_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(recommendations, f, indent=2, default=str)

    logger.info(f"Recommendations exported to {filename}")
    return filename

# ========== SINGLETON INSTANCE ==========

_system_instance = None

def get_recommendation_system() -> ExamRecommendationSystem:
    """Get singleton instance of recommendation system"""
    global _system_instance
    if _system_instance is None:
        _system_instance = ExamRecommendationSystem()
    return _system_instance

# ========== TEST FUNCTION ==========

def test_backend():
    """Test the backend system"""
    print("🧪 Testing Streamlined Exam Recommendation System Backend...")
    print("=" * 60)

    # Initialize system
    system = get_recommendation_system()

    # Test different student profiles
    test_profiles = [
        {
            "name": "Class 12 Science PCM student",
            "data": {
                "current_class": "12",
                "stream": Stream.SCIENCE_PCM.value,
                "state": "Maharashtra",
                "board": "State Board",
                "scholarship_needed": False,
                "percentage": 85.5,
                "age": 17
            }
        },
        {
            "name": "Class 10 student",
            "data": {
                "current_class": "10",
                "stream": "All",
                "state": "Delhi",
                "board": "CBSE",
                "scholarship_needed": True,
                "percentage": 92.0,
                "age": 15
            }
        },
        {
            "name": "UG Commerce student",
            "data": {
                "current_class": "UG",
                "stream": Stream.COMMERCE.value,
                "state": "All-India",
                "board": "N/A",
                "scholarship_needed": False,
                "percentage": 78.0,
                "age": 22
            }
        }
    ]

    for profile in test_profiles:
        print(f"\n📋 Testing: {profile['name']}")
        print("-" * 40)

        # Validate data
        is_valid, errors = validate_student_data(profile['data'])
        if not is_valid:
            print(f"❌ Validation errors: {errors}")
            continue

        # Process query
        print("Processing student query...")
        result = system.process_student_query(**profile['data'])

        # Display results
        print(f"✅ Successfully processed!")
        print(f"📊 Eligible exams found: {result['eligible_exams_count']}")
        
        if result['government_exams']:
            print(f"🎯 Top 3 exam recommendations:")
            for i, exam in enumerate(result['government_exams'][:3], 1):
                print(f"  {i}. {exam['name']} ({exam['exam_category']})")
                print(f"     Stream: {exam['stream']}, Level: {exam['level']}, Eligible from: {exam['eligible_from']}")
                if exam['last_date']:
                    print(f"     Last Date: {exam['last_date'][:10]}")
                print()

    # Final summary
    print("\n" + "=" * 60)
    print("✨ Backend test completed successfully!")
    
    # Show database statistics
    db_stats = system.govt_db.get_exams_statistics()
    print(f"\n📊 Database Statistics:")
    print(f"  Total Exams: {db_stats['total_exams']}")
    print(f"  By Level: {db_stats['by_level']}")
    print(f"  By Category: {len(db_stats['by_category'])} categories")
    print(f"  Scholarship Exams: {db_stats['scholarship_exams']}")

# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    # Run test if executed directly
    test_backend()