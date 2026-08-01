"""Quick DB check for video status."""
import asyncio
from backend.app.core.database import AsyncSessionLocal
from backend.app.models.video import Video
from sqlalchemy import select, text


async def check():
    async with AsyncSessionLocal() as db:
        # Check video 19
        r = await db.execute(select(Video).where(Video.id == 19))
        v = r.scalar_one_or_none()
        if v:
            print(f"Video 19: status={v.status}, scene={v.scene}, error={v.error_message}")
        else:
            print("Video 19: not found")

        # Check all USPCA videos
        r = await db.execute(text("SELECT id, status, scene, error_message FROM videos WHERE scene = 'uspca_patrol' ORDER BY id DESC LIMIT 5"))
        rows = r.all()
        print(f"\nUSPCA videos (last 5):")
        for row in rows:
            print(f"  id={row[0]}, status={row[1]}, scene={row[2]}, error={row[3]}")


asyncio.run(check())
