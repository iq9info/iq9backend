import os
import json
import asyncpg
from fastapi import FastAPI, BackgroundTasks, HTTPException
from passlib.context import CryptContext
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from app.models.schemas import UserRegister, UserLogin, PasswordUpdate, OnboardingRequest
from app.agents.agent1_synthesizer import Agent1Synthesizer
from app.agents.agent2_ingestor import Agent2Ingestor

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global db_pool
    # Supabase (Cloud Postgres) requires robust connection pooling
    db_pool = await asyncpg.create_pool(
        os.getenv("DATABASE_URL"),
        min_size=1,
        max_size=10,
        command_timeout=60,
        max_inactive_connection_lifetime=300
    )
    yield
    # Shutdown
    if db_pool:
        await db_pool.close()

app = FastAPI(title="iQ9 PRO LangGraph Engine", lifespan=lifespan)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
db_pool = None

# Initialize LangGraph Agent 1 Synthesizer
agent1 = Agent1Synthesizer()

@app.get("/")
async def root():
    return {"status": "SUCCESS", "message": "iQ9 PRO LangGraph Engine is running"}

STANDARD_MODULES = [
    "AI Fundamentals", "Machine Learning", "Deep Learning", "Advanced AI",
    "LLM Fundamentals", "LLM Advanced", "LLM Ops", "Prompt Engineering",
    "RAG", "MCP (Model Context Protocol)", "Agentic AI", "Multimodal Frameworks",
    "AI Ethics & Security"
]

@app.post("/api/v1/auth/signup")
async def signup(user: UserRegister):
    hashed_pwd = pwd_context.hash(user.password)
    async with db_pool.acquire() as conn:
        user_id = await conn.fetchval(
            """
            INSERT INTO users (email, password_hash, full_name, current_role, experience_years)
            VALUES ($1, $2, $3, $4, $5) RETURNING id
            """,
            user.email, hashed_pwd, user.full_name, user.current_role, user.experience_years
        )
        return {"status": "SUCCESS", "user_id": str(user_id)}

@app.post("/api/v1/auth/login")
async def login(credentials: UserLogin):
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT id, password_hash, is_pro_subscriber FROM users WHERE email = $1", credentials.email)
        if not row or not pwd_context.verify(credentials.password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return {"status": "SUCCESS", "user_id": str(row["id"]), "is_pro": row["is_pro_subscriber"]}

@app.post("/api/v1/user/update-password")
async def update_password(payload: PasswordUpdate):
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT password_hash FROM users WHERE id = $1", payload.user_id)
        if not row or not pwd_context.verify(payload.old_password, row["password_hash"]):
            raise HTTPException(status_code=400, detail="Invalid old password")
        
        new_hash = pwd_context.hash(payload.new_password)
        await conn.execute("UPDATE users SET password_hash = $1 WHERE id = $2", new_hash, payload.user_id)
        return {"status": "SUCCESS", "message": "Password updated successfully"}

# LangGraph Execution Task
async def generate_curriculum_task(user_id: str, request: OnboardingRequest):
    async with db_pool.acquire() as conn:
        user = await conn.fetchrow("SELECT full_name, current_role, experience_years FROM users WHERE id = $1", user_id)
        user_profile = dict(user)
        user_profile.update({
            "target_role": request.target_role,
            "proficiency_level": request.proficiency_level
        })

    agent2 = Agent2Ingestor(db_pool)
    all_topics = STANDARD_MODULES + request.custom_topics

    for seq, topic in enumerate(all_topics, start=1):
        is_custom = topic in request.custom_topics
        
        # Invokes LangGraph State Workflow
        mod_schema = agent1.generate_module_content(topic, user_profile, seq, is_custom)
        
        # Ingest validated graph output into PostgreSQL via Agent 2
        await agent2.ingest_and_sync_module(user_id, mod_schema)

@app.post("/api/v1/onboarding/submit")
async def submit_onboarding(request: OnboardingRequest, background_tasks: BackgroundTasks):
    async with db_pool.acquire() as conn:
        await conn.execute("UPDATE users SET proficiency_level = $1 WHERE id = $2", request.proficiency_level, request.user_id)
    
    background_tasks.add_task(generate_curriculum_task, request.user_id, request)
    return {"status": "PROCESSING", "message": "LangGraph stateful workflow started in background."}

@app.get("/api/v1/curriculum/{user_id}")
async def get_curriculum(user_id: str):
    async with db_pool.acquire() as conn:
        modules = await conn.fetch("SELECT * FROM modules WHERE user_id = $1 ORDER BY sequence_order", user_id)
        result = []
        for mod in modules:
            mod_dict = dict(mod)
            mod_dict["submodules"] = []
            submodules = await conn.fetch("SELECT * FROM submodules WHERE module_id = $1 ORDER BY sequence_order", mod["id"])
            for sub in submodules:
                sub_dict = dict(sub)
                sub_dict["content_items"] = []
                content = await conn.fetch("SELECT * FROM submodule_content_items WHERE submodule_id = $1 ORDER BY item_index", sub["id"])
                for item in content:
                    item_dict = dict(item)
                    item_dict["content_payload"] = json.loads(item["content_payload"])
                    sub_dict["content_items"].append(item_dict)
                mod_dict["submodules"].append(sub_dict)

            mod_dict["assessment_questions"] = []
            questions = await conn.fetch("SELECT * FROM assessment_questions WHERE module_id = $1 ORDER BY question_index", mod["id"])
            for q in questions:
                q_dict = dict(q)
                q_dict["question_payload"] = json.loads(q["question_payload"])
                mod_dict["assessment_questions"].append(q_dict)

            result.append(mod_dict)
        return result

@app.get("/api/v1/leaderboard")
async def get_leaderboard():
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT user_id, full_name, current_role, total_xp, current_streak, global_rank FROM global_leaderboard LIMIT 50")
        return [dict(r) for r in rows]
