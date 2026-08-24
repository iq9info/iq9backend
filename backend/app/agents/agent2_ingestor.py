import json
import asyncpg
from app.models.schemas import ModuleSchema

class Agent2Ingestor:
    def __init__(self, db_pool: asyncpg.Pool):
        self.pool = db_pool

    async def ingest_and_sync_module(self, user_id: str, module_data: ModuleSchema) -> int:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # 1. Insert Module
                module_id = await conn.fetchval(
                    """
                    INSERT INTO modules (user_id, title, sequence_order, is_custom_topic)
                    VALUES ($1, $2, $3, $4) RETURNING id
                    """,
                    user_id, module_data.module_title, module_data.sequence_order, module_data.is_custom_topic
                )

                # 2. Insert Submodules & Content Items
                for sub in module_data.submodules:
                    submodule_id = await conn.fetchval(
                        """
                        INSERT INTO submodules (module_id, title, sequence_order)
                        VALUES ($1, $2, $3) RETURNING id
                        """,
                        module_id, sub.title, sub.sequence_order
                    )

                    for item in sub.content_items:
                        await conn.execute(
                            """
                            INSERT INTO submodule_content_items 
                            (submodule_id, item_index, content_type, title, content_payload, difficulty_level)
                            VALUES ($1, $2, $3, $4, $5, $6)
                            """,
                            submodule_id, item.item_index, item.content_type, item.title,
                            json.dumps(item.content_payload), item.difficulty_level
                        )

                # 3. Insert Assessment Questions
                for q in module_data.assessment_questions:
                    await conn.execute(
                        """
                        INSERT INTO assessment_questions 
                        (module_id, question_index, question_type, question_text, question_payload, explanation, difficulty_level)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """,
                        module_id, q.question_index, q.question_type, q.question_text,
                        json.dumps(q.question_payload), q.explanation, q.difficulty_level
                    )

                return module_id