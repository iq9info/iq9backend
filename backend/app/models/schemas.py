from pydantic import BaseModel, Field, EmailStr
from typing import List, Dict, Any, Optional

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    current_role: str
    experience_years: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class PasswordUpdate(BaseModel):
    user_id: str
    old_password: str
    new_password: str

class ContentItemSchema(BaseModel):
    item_index: int = Field(ge=1, le=50)
    content_type: str
    title: str
    content_payload: Dict[str, Any]
    difficulty_level: str

class AssessmentQuestionSchema(BaseModel):
    question_index: int = Field(ge=1, le=50)
    question_type: str
    question_text: str
    question_payload: Dict[str, Any]
    explanation: str
    difficulty_level: str

class SubmoduleSchema(BaseModel):
    title: str
    sequence_order: int
    content_items: List[ContentItemSchema]

class ModuleSchema(BaseModel):
    module_title: str
    sequence_order: int
    is_custom_topic: bool = False
    submodules: List[SubmoduleSchema]
    assessment_questions: List[AssessmentQuestionSchema]

class OnboardingRequest(BaseModel):
    user_id: str
    target_role: str
    proficiency_level: str
    custom_topics: List[str] = []