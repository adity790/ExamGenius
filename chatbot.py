"""
Chatbot Module: Groq LLM powered Personalized Guidance
Used as an advisory layer on top of ML recommendations
"""

from groq import Groq

# ================= GROQ CONFIG =================

GROQ_API_KEY = "PASTE_YOUR_GROQ_API_KEY_HERE"

client = Groq(api_key="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")


# ================= MAIN CHATBOT FUNCTION =================

def generate_personalized_guidance(student_profile, recommendations):
    """
    student_profile : dict (from session_state.student_profile)
    recommendations : dict (ML enhanced result from backend)

    Returns: string (LLM-generated guidance)
    """

    # ---------------- Extract Student Info ----------------
    current_class = student_profile.get("current_class")
    stream = student_profile.get("stream")
    state = student_profile.get("state")
    board = student_profile.get("board")
    scholarship = "Yes" if student_profile.get("scholarship_needed") else "No"
    age = student_profile.get("age")
    percentage = student_profile.get("percentage")

    # ---------------- Extract Top Exams ----------------
    gov_exams = recommendations.get("government_exams", [])
    top_exams = gov_exams[:5]  # only top 5 for clarity

    exam_summary = []
    for idx, exam in enumerate(top_exams, start=1):
        exam_summary.append(
            f"{idx}. {exam.get('name', 'Unknown Exam')} "
            f"(Level: {exam.get('level')}, "
            f"Eligibility: {exam.get('eligible_from')}+, "
            f"ML Score: {round(exam.get('ml_enhanced_score', 0.5), 2)})"
        )

    exam_text = "\n".join(exam_summary) if exam_summary else "No exams found."

    # ---------------- Prompt Engineering ----------------
    prompt = f"""
You are an expert academic mentor for Indian students.

STUDENT PROFILE:
- Current Class/Level: {current_class}
- Stream: {stream}
- State: {state}
- Board: {board}
- Scholarship Needed: {scholarship}
- Age: {age}
- Percentage/CGPA: {percentage}

TOP ML-RECOMMENDED EXAMS:
{exam_text}

TASK:
1. Give personalized preparation advice
2. Suggest subject-wise strategy
3. Mention common mistakes students at this level make
4. Provide a realistic 3-month preparation roadmap
5. Keep tone motivating and practical (not generic)

IMPORTANT:
- Do NOT invent exams
- Base advice strictly on given profile
- Indian education context only
"""

    # ---------------- Groq Call ----------------
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful academic advisor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6
    )

    return response.choices[0].message.content
