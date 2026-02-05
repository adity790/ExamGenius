"""
Database Module: Data Persistence Layer for Exam Recommendation System
Version: 2.0 - Simplified without career_goal and AI tracking
"""

import pandas as pd
import sqlite3
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import csv
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== DATA MODELS ==========

@dataclass
class UniversityExam:
    """University/Institution exam data model"""
    university_name: str
    exam_name: str
    exam_level: str
    eligible_from: str
    stream: str
    age_limit: str
    exam_mode: str
    scholarship: str
    last_date: str
    contact_email: str
    website: str
    registration_date: str
    exam_id: Optional[str] = None
    min_percentage: Optional[float] = None
    application_fee: Optional[float] = None
    syllabus_link: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    
    def __post_init__(self):
        """Generate unique exam ID if not provided"""
        if not self.exam_id:
            unique_string = f"{self.university_name}_{self.exam_name}_{self.registration_date}"
            self.exam_id = hashlib.md5(unique_string.encode()).hexdigest()[:12]
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate exam data"""
        errors = []
        
        required_fields = [
            ('university_name', self.university_name),
            ('exam_name', self.exam_name),
            ('exam_level', self.exam_level),
            ('eligible_from', self.eligible_from),
            ('stream', self.stream),
            ('contact_email', self.contact_email)
        ]
        
        for field_name, value in required_fields:
            if not value or str(value).strip() == '':
                errors.append(f"Missing required field: {field_name}")
        
        # Email validation
        if self.contact_email and '@' not in self.contact_email:
            errors.append("Invalid email format")
        
        # Date validation
        try:
            datetime.strptime(self.last_date, '%Y-%m-%d')
        except (ValueError, TypeError):
            errors.append("Invalid date format. Use YYYY-MM-DD")
        
        # Percentage validation
        if self.min_percentage and (self.min_percentage < 0 or self.min_percentage > 100):
            errors.append("Percentage must be between 0 and 100")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)

@dataclass
class StudentQuery:
    """Student query tracking for analytics"""
    query_id: str
    student_class: str
    stream: str
    state: str
    timestamp: str
    eligible_exams_count: int
    
    def to_dict(self) -> Dict:
        return asdict(self)

# ========== DATABASE MANAGER ==========

class ExamDatabaseManager:
    """
    Main database manager handling both CSV and SQLite operations
    """
    
    def __init__(self, data_dir: str = "data", use_sqlite: bool = False):
        """
        Initialize database manager
        
        Args:
            data_dir: Directory to store data files
            use_sqlite: If True, use SQLite database; else use CSV
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.use_sqlite = use_sqlite
        self.csv_file = self.data_dir / "university_exams.csv"
        self.sqlite_file = self.data_dir / "exams.db"
        
        # Initialize data storage
        if use_sqlite:
            self._init_sqlite()
        else:
            self._init_csv()
        
        # Analytics storage
        self.queries_file = self.data_dir / "student_queries.csv"
        self._init_analytics()
        
        logger.info(f"Database initialized: {'SQLite' if use_sqlite else 'CSV'}")
    
    # ===== INITIALIZATION METHODS =====
    
    def _init_csv(self) -> None:
        """Initialize CSV file with proper columns"""
        if not self.csv_file.exists():
            columns = [
                'exam_id', 'university_name', 'exam_name', 'exam_level',
                'eligible_from', 'stream', 'age_limit', 'exam_mode',
                'scholarship', 'last_date', 'contact_email', 'website',
                'registration_date', 'min_percentage', 'application_fee',
                'syllabus_link', 'state', 'city'
            ]
            
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(columns)
            
            logger.info(f"Created new CSV file: {self.csv_file}")
    
    def _init_sqlite(self) -> None:
        """Initialize SQLite database with tables"""
        try:
            conn = sqlite3.connect(self.sqlite_file)
            cursor = conn.cursor()
            
            # Create university exams table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS university_exams (
                    exam_id TEXT PRIMARY KEY,
                    university_name TEXT NOT NULL,
                    exam_name TEXT NOT NULL,
                    exam_level TEXT NOT NULL,
                    eligible_from TEXT NOT NULL,
                    stream TEXT NOT NULL,
                    age_limit TEXT,
                    exam_mode TEXT,
                    scholarship TEXT,
                    last_date TEXT NOT NULL,
                    contact_email TEXT NOT NULL,
                    website TEXT,
                    registration_date TEXT NOT NULL,
                    min_percentage REAL,
                    application_fee REAL,
                    syllabus_link TEXT,
                    state TEXT,
                    city TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create student queries table (analytics)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS student_queries (
                    query_id TEXT PRIMARY KEY,
                    student_class TEXT NOT NULL,
                    stream TEXT NOT NULL,
                    state TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    eligible_exams_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_stream ON university_exams(stream)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_exam_level ON university_exams(exam_level)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_state ON university_exams(state)')
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error initializing SQLite: {e}")
    
    def _init_analytics(self) -> None:
        """Initialize analytics storage"""
        if not self.queries_file.exists():
            columns = [
                'query_id', 'student_class', 'stream',
                'state', 'timestamp', 'eligible_exams_count'
            ]
            
            with open(self.queries_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(columns)
    
    # ===== UNIVERSITY EXAM OPERATIONS =====
    
    def add_exam(self, exam_data: Dict) -> Tuple[bool, str, Optional[str]]:
        """
        Add a new university exam to database
        
        Returns:
            Tuple of (success, message, exam_id)
        """
        try:
            # Add registration date
            exam_data['registration_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Ensure all fields are present
            for field in ['min_percentage', 'application_fee', 'syllabus_link', 'state', 'city']:
                if field not in exam_data:
                    exam_data[field] = None
            
            # Create exam object
            exam = UniversityExam(**exam_data)
            
            # Validate data
            is_valid, errors = exam.validate()
            if not is_valid:
                return False, f"Validation failed: {', '.join(errors)}", None
            
            # Save to database
            if self.use_sqlite:
                success = self._save_to_sqlite(exam)
            else:
                success = self._save_to_csv(exam)
            
            if success:
                logger.info(f"Added exam: {exam.exam_name} by {exam.university_name}")
                return True, "Exam registered successfully", exam.exam_id
            else:
                return False, "Failed to save exam", None
                
        except Exception as e:
            logger.error(f"Error adding exam: {e}")
            return False, f"Error: {str(e)}", None
    
    def _save_to_csv(self, exam: UniversityExam) -> bool:
        """Save exam to CSV file"""
        try:
            # Read existing data
            try:
                if self.csv_file.exists() and os.path.getsize(self.csv_file) > 0:
                    df_existing = pd.read_csv(self.csv_file)
                else:
                    df_existing = pd.DataFrame()
            except Exception:
                df_existing = pd.DataFrame()
            
            # Create new row
            exam_dict = exam.to_dict()
            new_row = pd.DataFrame([exam_dict])
            
            # Append and save
            if df_existing.empty:
                df_updated = new_row
            else:
                df_updated = pd.concat([df_existing, new_row], ignore_index=True)
            
            df_updated.to_csv(self.csv_file, index=False, encoding='utf-8')
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
            return False
    
    def _save_to_sqlite(self, exam: UniversityExam) -> bool:
        """Save exam to SQLite database"""
        try:
            conn = sqlite3.connect(self.sqlite_file)
            cursor = conn.cursor()
            
            exam_dict = exam.to_dict()
            columns = ', '.join(exam_dict.keys())
            placeholders = ', '.join(['?' for _ in exam_dict])
            values = list(exam_dict.values())
            
            query = f"INSERT OR REPLACE INTO university_exams ({columns}) VALUES ({placeholders})"
            cursor.execute(query, values)
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving to SQLite: {e}")
            return False
    
    def get_exams(self, filters: Dict = None) -> pd.DataFrame:
        """
        Get exams with optional filtering
        
        Args:
            filters: Dictionary of filter conditions
                    Example: {'stream': 'Science (PCM)', 'exam_level': 'UG'}
        
        Returns:
            DataFrame of matching exams
        """
        try:
            if self.use_sqlite:
                df = self._get_from_sqlite(filters)
            else:
                df = self._get_from_csv(filters)
            
            # Sort by last_date (ascending - soonest first)
            if not df.empty and 'last_date' in df.columns:
                df['last_date'] = pd.to_datetime(df['last_date'], errors='coerce')
                df = df.sort_values('last_date')
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting exams: {e}")
            return pd.DataFrame()
    
    def _get_from_csv(self, filters: Dict = None) -> pd.DataFrame:
        """Get exams from CSV with filtering"""
        try:
            if not self.csv_file.exists() or os.path.getsize(self.csv_file) == 0:
                return pd.DataFrame()
            
            df = pd.read_csv(self.csv_file, encoding='utf-8')
            
            if df.empty:
                return df
            
            # Apply filters if provided
            if filters:
                for key, value in filters.items():
                    if key in df.columns:
                        if isinstance(value, list):
                            df = df[df[key].isin(value)]
                        else:
                            df = df[df[key] == value]
            
            return df
            
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            return pd.DataFrame()
    
    def _get_from_sqlite(self, filters: Dict = None) -> pd.DataFrame:
        """Get exams from SQLite with filtering"""
        try:
            conn = sqlite3.connect(self.sqlite_file)
            
            # Build query
            query = "SELECT * FROM university_exams WHERE 1=1"
            params = []
            
            if filters:
                for key, value in filters.items():
                    if isinstance(value, list):
                        placeholders = ', '.join(['?' for _ in value])
                        query += f" AND {key} IN ({placeholders})"
                        params.extend(value)
                    else:
                        query += f" AND {key} = ?"
                        params.append(value)
            
            query += " ORDER BY last_date ASC"
            
            if params:
                df = pd.read_sql_query(query, conn, params=params)
            else:
                df = pd.read_sql_query(query, conn)
                
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Error querying SQLite: {e}")
            return pd.DataFrame()
    
    def get_exams_for_student(self, student_class: str, stream: str) -> pd.DataFrame:
        """
        Get exams eligible for a specific student
        
        Args:
            student_class: Student's current class/level
            stream: Student's academic stream
        
        Returns:
            DataFrame of eligible exams
        """
        # Map student class to eligibility
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
        
        try:
            # Get all exams for the stream
            exams = self.get_exams({'stream': [stream, 'All']})
            
            if exams.empty:
                return exams
            
            # Filter by eligibility
            if student_class in eligibility_map:
                eligible_levels = eligibility_map[student_class]
                exams = exams[exams['eligible_from'].isin(eligible_levels)]
            
            return exams
            
        except Exception as e:
            logger.error(f"Error filtering exams for student: {e}")
            return pd.DataFrame()
    
    def delete_exam(self, exam_id: str) -> Tuple[bool, str]:
        """Delete an exam by ID"""
        try:
            if self.use_sqlite:
                conn = sqlite3.connect(self.sqlite_file)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM university_exams WHERE exam_id = ?", (exam_id,))
                affected = cursor.rowcount
                conn.commit()
                conn.close()
                success = affected > 0
            else:
                # CSV deletion
                if not self.csv_file.exists():
                    return False, "No exams database found"
                
                df = pd.read_csv(self.csv_file)
                original_len = len(df)
                df = df[df['exam_id'] != exam_id]
                success = len(df) < original_len
                if success:
                    df.to_csv(self.csv_file, index=False)
            
            if success:
                logger.info(f"Deleted exam: {exam_id}")
                return True, "Exam deleted successfully"
            else:
                return False, "Exam not found"
                
        except Exception as e:
            logger.error(f"Error deleting exam: {e}")
            return False, f"Error: {str(e)}"
    
    def update_exam(self, exam_id: str, updates: Dict) -> Tuple[bool, str]:
        """Update an existing exam"""
        try:
            # Get current exam data
            if self.use_sqlite:
                conn = sqlite3.connect(self.sqlite_file)
                df = pd.read_sql_query(
                    "SELECT * FROM university_exams WHERE exam_id = ?",
                    conn,
                    params=(exam_id,)
                )
                conn.close()
            else:
                if not self.csv_file.exists():
                    return False, "No exams database found"
                df = pd.read_csv(self.csv_file)
                df = df[df['exam_id'] == exam_id]
            
            if df.empty:
                return False, "Exam not found"
            
            # Update fields
            current_data = df.iloc[0].to_dict()
            current_data.update(updates)
            
            # Re-validate
            exam = UniversityExam(**current_data)
            is_valid, errors = exam.validate()
            if not is_valid:
                return False, f"Validation failed: {', '.join(errors)}"
            
            # Save updated data
            if self.use_sqlite:
                conn = sqlite3.connect(self.sqlite_file)
                cursor = conn.cursor()
                
                set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
                query = f"UPDATE university_exams SET {set_clause} WHERE exam_id = ?"
                values = list(updates.values()) + [exam_id]
                
                cursor.execute(query, values)
                conn.commit()
                conn.close()
            else:
                df_all = pd.read_csv(self.csv_file)
                idx = df_all[df_all['exam_id'] == exam_id].index
                if len(idx) > 0:
                    for key, value in updates.items():
                        df_all.at[idx[0], key] = value
                    df_all.to_csv(self.csv_file, index=False)
            
            logger.info(f"Updated exam: {exam_id}")
            return True, "Exam updated successfully"
            
        except Exception as e:
            logger.error(f"Error updating exam: {e}")
            return False, f"Error: {str(e)}"
    
    # ===== ANALYTICS OPERATIONS =====
    
    def log_student_query(self, query_data: Dict) -> bool:
        """Log student query for analytics"""
        try:
            query_data['timestamp'] = datetime.now().isoformat()
            
            query_data['query_id'] = hashlib.md5(
                json.dumps(query_data, sort_keys=True).encode()
            ).hexdigest()[:10]
            
            query = StudentQuery(**query_data)
            
            if self.use_sqlite:
                conn = sqlite3.connect(self.sqlite_file)
                cursor = conn.cursor()
                
                query_dict = query.to_dict()
                columns = ', '.join(query_dict.keys())
                placeholders = ', '.join(['?' for _ in query_dict])
                values = list(query_dict.values())
                
                cursor.execute(
                    f"INSERT INTO student_queries ({columns}) VALUES ({placeholders})",
                    values
                )
                conn.commit()
                conn.close()
            else:
                # Save to CSV
                new_row = pd.DataFrame([query.to_dict()])
                
                if self.queries_file.exists() and os.path.getsize(self.queries_file) > 0:
                    df_existing = pd.read_csv(self.queries_file)
                    df_updated = pd.concat([df_existing, new_row], ignore_index=True)
                else:
                    df_updated = new_row
                
                df_updated.to_csv(self.queries_file, index=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Error logging query: {e}")
            return False
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get system analytics"""
        try:
            if self.use_sqlite:
                conn = sqlite3.connect(self.sqlite_file)
                
                # Total exams
                total_exams = pd.read_sql_query(
                    "SELECT COUNT(*) as count FROM university_exams",
                    conn
                ).iloc[0]['count'] if not pd.read_sql_query(
                    "SELECT COUNT(*) as count FROM university_exams", conn
                ).empty else 0
                
                # Exams by stream
                stream_df = pd.read_sql_query(
                    "SELECT stream, COUNT(*) as count FROM university_exams GROUP BY stream",
                    conn
                )
                by_stream = []
                if not stream_df.empty:
                    for _, row in stream_df.iterrows():
                        by_stream.append({
                            'stream': row['stream'],
                            'count': int(row['count'])
                        })
                
                # Exams by level
                level_df = pd.read_sql_query(
                    "SELECT exam_level, COUNT(*) as count FROM university_exams GROUP BY exam_level",
                    conn
                )
                by_level = []
                if not level_df.empty:
                    for _, row in level_df.iterrows():
                        by_level.append({
                            'exam_level': row['exam_level'],
                            'count': int(row['count'])
                        })
                
                # Recent queries
                recent_queries = pd.read_sql_query(
                    "SELECT * FROM student_queries ORDER BY timestamp DESC LIMIT 50",
                    conn
                ) if not pd.read_sql_query(
                    "SELECT * FROM student_queries ORDER BY timestamp DESC LIMIT 50", conn
                ).empty else pd.DataFrame()
                
                conn.close()
                
            else:
                # CSV analytics
                try:
                    if self.csv_file.exists() and os.path.getsize(self.csv_file) > 0:
                        exams_df = pd.read_csv(self.csv_file)
                        total_exams = len(exams_df)
                        
                        if not exams_df.empty:
                            by_stream = []
                            if 'stream' in exams_df.columns:
                                stream_counts = exams_df['stream'].value_counts()
                                for stream, count in stream_counts.items():
                                    by_stream.append({
                                        'stream': str(stream),
                                        'count': int(count)
                                    })
                            
                            by_level = []
                            if 'exam_level' in exams_df.columns:
                                level_counts = exams_df['exam_level'].value_counts()
                                for level, count in level_counts.items():
                                    by_level.append({
                                        'exam_level': str(level),
                                        'count': int(count)
                                    })
                        else:
                            by_stream = []
                            by_level = []
                    else:
                        total_exams = 0
                        by_stream = []
                        by_level = []
                    
                    # Read queries if file exists
                    if self.queries_file.exists() and os.path.getsize(self.queries_file) > 0:
                        recent_queries = pd.read_csv(self.queries_file)
                        recent_queries = recent_queries.tail(50)
                    else:
                        recent_queries = pd.DataFrame()
                        
                except Exception as e:
                    logger.error(f"Error in CSV analytics: {e}")
                    total_exams = 0
                    by_stream = []
                    by_level = []
                    recent_queries = pd.DataFrame()
            
            # Calculate statistics
            today = datetime.now().date()
            upcoming_exams = 0
            
            if self.use_sqlite:
                conn = sqlite3.connect(self.sqlite_file)
                upcoming = pd.read_sql_query(
                    "SELECT COUNT(*) as count FROM university_exams WHERE last_date >= ?",
                    conn,
                    params=(today.isoformat(),)
                )
                conn.close()
                upcoming_exams = upcoming.iloc[0]['count'] if not upcoming.empty else 0
            else:
                try:
                    if self.csv_file.exists() and os.path.getsize(self.csv_file) > 0:
                        exams_df = pd.read_csv(self.csv_file)
                        if not exams_df.empty and 'last_date' in exams_df.columns:
                            exams_df['last_date'] = pd.to_datetime(exams_df['last_date'], errors='coerce')
                            upcoming_exams = len(exams_df[exams_df['last_date'].dt.date >= today])
                except Exception:
                    upcoming_exams = 0
            
            return {
                'total_exams': int(total_exams),
                'upcoming_exams': int(upcoming_exams),
                'exams_by_stream': by_stream,
                'exams_by_level': by_level,
                'recent_queries': recent_queries.to_dict('records') if not recent_queries.empty else [],
                'generated_at': datetime.now().isoformat(),
                'database_type': 'SQLite' if self.use_sqlite else 'CSV'
            }
            
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return {
                'total_exams': 0,
                'upcoming_exams': 0,
                'exams_by_stream': [],
                'exams_by_level': [],
                'recent_queries': [],
                'generated_at': datetime.now().isoformat(),
                'database_type': 'SQLite' if self.use_sqlite else 'CSV',
                'error': str(e)
            }
    
    # ===== EXPORT/IMPORT OPERATIONS =====
    
    def export_to_json(self, filepath: str = None) -> str:
        """Export all exams to JSON file"""
        try:
            if not filepath:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = self.data_dir / f"exams_export_{timestamp}.json"
            
            df = self.get_exams()
            
            if df.empty:
                data = {"exams": [], "export_date": datetime.now().isoformat()}
            else:
                data = {
                    "exams": df.to_dict('records'),
                    "export_date": datetime.now().isoformat(),
                    "total_exams": len(df)
                }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported {len(df)} exams to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            return ""
    
    def import_from_json(self, filepath: str) -> Tuple[bool, str, int]:
        """Import exams from JSON file"""
        try:
            if not os.path.exists(filepath):
                return False, "File not found", 0
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'exams' not in data:
                return False, "Invalid JSON format: missing 'exams' key", 0
            
            imported_count = 0
            for exam_data in data['exams']:
                success, message, _ = self.add_exam(exam_data)
                if success:
                    imported_count += 1
            
            return True, f"Imported {imported_count} exams", imported_count
            
        except Exception as e:
            logger.error(f"Error importing from JSON: {e}")
            return False, f"Error: {str(e)}", 0
    
    def backup_database(self) -> str:
        """Create a backup of the database"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if self.use_sqlite:
                backup_file = self.data_dir / f"exams_backup_{timestamp}.db"
                import shutil
                if self.sqlite_file.exists():
                    shutil.copy2(self.sqlite_file, backup_file)
                else:
                    return ""
            else:
                backup_file = self.data_dir / f"exams_backup_{timestamp}.csv"
                import shutil
                if self.csv_file.exists():
                    shutil.copy2(self.csv_file, backup_file)
                else:
                    return ""
            
            logger.info(f"Created backup: {backup_file}")
            return str(backup_file)
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return ""
    
    # ===== UTILITY METHODS =====
    
    def get_exam_by_id(self, exam_id: str) -> Optional[Dict]:
        """Get single exam by ID"""
        try:
            df = self.get_exams({'exam_id': exam_id})
            if not df.empty:
                return df.iloc[0].to_dict()
            return None
        except Exception:
            return None
    
    def get_unique_streams(self) -> List[str]:
        """Get list of unique streams in database"""
        try:
            df = self.get_exams()
            if not df.empty and 'stream' in df.columns:
                streams = sorted(df['stream'].unique().tolist())
                return [str(s) for s in streams]
            return []
        except Exception:
            return []
    
    def get_unique_universities(self) -> List[str]:
        """Get list of unique universities"""
        try:
            df = self.get_exams()
            if not df.empty and 'university_name' in df.columns:
                universities = sorted(df['university_name'].unique().tolist())
                return [str(u) for u in universities]
            return []
        except Exception:
            return []
    
    def search_exams(self, search_term: str, field: str = "exam_name") -> pd.DataFrame:
        """Search exams by text in specified field"""
        try:
            df = self.get_exams()
            if df.empty:
                return df
            
            if field in df.columns:
                mask = df[field].astype(str).str.contains(
                    search_term, case=False, na=False
                )
                return df[mask]
            
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()
    
    def get_stats_summary(self) -> Dict:
        """Get quick stats summary"""
        try:
            analytics = self.get_analytics()
            
            return {
                'total_exams': analytics.get('total_exams', 0),
                'upcoming_exams': analytics.get('upcoming_exams', 0),
                'unique_streams': len(self.get_unique_streams()),
                'unique_universities': len(self.get_unique_universities()),
                'database_file': str(self.sqlite_file if self.use_sqlite else self.csv_file)
            }
        except Exception:
            return {
                'total_exams': 0,
                'upcoming_exams': 0,
                'unique_streams': 0,
                'unique_universities': 0,
                'database_file': str(self.sqlite_file if self.use_sqlite else self.csv_file)
            }

# ========== SINGLETON INSTANCE ==========

_database_instance = None

def get_database_manager(use_sqlite: bool = False) -> ExamDatabaseManager:
    """Get singleton instance of database manager"""
    global _database_instance
    if _database_instance is None:
        _database_instance = ExamDatabaseManager(use_sqlite=use_sqlite)
    return _database_instance

# ========== TEST FUNCTION ==========

def test_database():
    """Test the database module"""
    print("🧪 Testing Exam Database Module...")
    
    # Initialize database (CSV mode)
    db = ExamDatabaseManager(use_sqlite=False)
    
    # Test 1: Add sample exams
    print("\n1. Adding sample exams...")
    
    sample_exams = [
        {
            "university_name": "University of Mumbai",
            "exam_name": "MU-CET Engineering 2025",
            "exam_level": "UG",
            "eligible_from": "12",
            "stream": "Science (PCM)",
            "age_limit": "None",
            "exam_mode": "Online",
            "scholarship": "Yes",
            "last_date": "2024-12-15",
            "contact_email": "admissions@mu.ac.in",
            "website": "https://mu.ac.in",
            "min_percentage": 60.0,
            "state": "Maharashtra",
            "city": "Mumbai"
        },
        {
            "university_name": "Delhi University",
            "exam_name": "DUET 2025",
            "exam_level": "UG",
            "eligible_from": "12",
            "stream": "All",
            "age_limit": "None",
            "exam_mode": "Online",
            "scholarship": "Yes",
            "last_date": "2025-01-31",
            "contact_email": "duet@du.ac.in",
            "website": "https://du.ac.in",
            "min_percentage": 50.0,
            "state": "Delhi",
            "city": "New Delhi"
        }
    ]
    
    for exam_data in sample_exams:
        success, message, exam_id = db.add_exam(exam_data)
        print(f"   ✓ {exam_data['exam_name']}: {message} (ID: {exam_id})")
    
    # Test 2: Get all exams
    print("\n2. Retrieving all exams...")
    all_exams = db.get_exams()
    print(f"   Total exams in database: {len(all_exams)}")
    
    # Test 3: Filter exams
    print("\n3. Filtering exams for Science (PCM) stream...")
    science_exams = db.get_exams({'stream': 'Science (PCM)'})
    print(f"   Science exams found: {len(science_exams)}")
    
    # Test 4: Student eligibility check
    print("\n4. Checking eligibility for Class 12 Science student...")
    student_exams = db.get_exams_for_student('12', 'Science (PCM)')
    print(f"   Eligible exams for student: {len(student_exams)}")
    
    # Test 5: Analytics
    print("\n5. Getting analytics...")
    analytics = db.get_analytics()
    print(f"   Database stats: {analytics.get('total_exams', 0)} total exams")
    print(f"   By stream: {len(analytics.get('exams_by_stream', []))} streams")
    
    # Test 6: Log student query
    print("\n6. Logging student query...")
    query_data = {
        "student_class": "12",
        "stream": "Science (PCM)",
        "state": "Maharashtra",
        "eligible_exams_count": 5
    }
    db.log_student_query(query_data)
    print("   ✓ Query logged successfully")
    
    # Test 7: Export to JSON
    print("\n7. Exporting to JSON...")
    export_file = db.export_to_json()
    if export_file:
        print(f"   ✓ Exported to: {export_file}")
    
    # Test 8: Get stats summary
    print("\n8. Stats summary:")
    stats = db.get_stats_summary()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✨ Database module test completed successfully!")

# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    # Run test if executed directly
    test_database()