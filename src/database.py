import sqlite3
import json
from datetime import datetime, timedelta
from typing import Optional


class ViolationDB:
    def __init__(self, db_path: str = "data/violations.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    image_path TEXT,
                    plate_number TEXT,
                    violation_types TEXT,
                    confidence_scores TEXT,
                    location TEXT,
                    annotated_image_path TEXT,
                    status TEXT DEFAULT 'pending'
                )
            """)

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def insert_violation(self, case_id: str, plate: Optional[str],
                         violations_list: list, image_path: str,
                         annotated_path: str,
                         location: Optional[str] = None) -> str:
        types = [v["violation_type"] for v in violations_list]
        scores = {v["violation_type"]: v["confidence"]
                  for v in violations_list}
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO violations
                    (case_id, timestamp, image_path, plate_number,
                     violation_types, confidence_scores, location,
                     annotated_image_path, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            """, (case_id, ts, image_path, plate or "",
                  json.dumps(types), json.dumps(scores),
                  location or "", annotated_path))
            conn.commit()
        finally:
            conn.close()
        return case_id

    def get_all(self, limit: int = 100) -> list:
        conn = self._get_conn()
        try:
            cursor = conn.execute("""
                SELECT * FROM violations
                ORDER BY id DESC LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def search_by_plate(self, plate_text: str) -> list:
        conn = self._get_conn()
        try:
            cursor = conn.execute("""
                SELECT * FROM violations
                WHERE plate_number LIKE ?
                ORDER BY id DESC
            """, (f"%{plate_text}%",))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def search_by_type(self, violation_type: str) -> list:
        conn = self._get_conn()
        try:
            cursor = conn.execute("""
                SELECT * FROM violations
                WHERE violation_types LIKE ?
                ORDER BY id DESC
            """, (f"%{violation_type}%",))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_stats(self) -> dict:
        conn = self._get_conn()
        try:
            total = conn.execute(
                "SELECT COUNT(*) FROM violations"
            ).fetchone()[0]

            by_type_raw = conn.execute(
                "SELECT violation_types FROM violations"
            ).fetchall()
            by_type = {}
            for row in by_type_raw:
                try:
                    types = json.loads(row[0])
                    for t in types:
                        by_type[t] = by_type.get(t, 0) + 1
                except (json.JSONDecodeError, TypeError):
                    pass

            seven_days_ago = (datetime.now() - timedelta(days=7)
                              ).strftime("%Y-%m-%d")
            daily_raw = conn.execute("""
                SELECT DATE(timestamp) as day, COUNT(*) as cnt
                FROM violations
                WHERE DATE(timestamp) >= ?
                GROUP BY day ORDER BY day
            """, (seven_days_ago,)).fetchall()
            daily_trend = {row[0]: row[1] for row in daily_raw}

            top_raw = conn.execute("""
                SELECT plate_number, COUNT(*) as cnt
                FROM violations
                WHERE plate_number != ''
                GROUP BY plate_number
                ORDER BY cnt DESC LIMIT 5
            """).fetchall()
            top_plates = {row[0]: row[1] for row in top_raw}

            return {
                "total_violations": total,
                "by_type": by_type,
                "daily_trend": daily_trend,
                "top_plates": top_plates,
            }
        finally:
            conn.close()

    def update_status(self, case_id: str, status: str):
        conn = self._get_conn()
        try:
            conn.execute("""
                UPDATE violations SET status = ?
                WHERE case_id = ?
            """, (status, case_id))
            conn.commit()
        finally:
            conn.close()


if __name__ == "__main__":
    db = ViolationDB()
    db.insert_violation(
        case_id="TEST_001",
        plate="MH12AB1234",
        violations_list=[
            {"violation_type": "HELMET_VIOLATION", "confidence": 0.89},
        ],
        image_path="data/test_images/sample.jpg",
        annotated_path="outputs/violations/TEST_001.jpg",
        location="Andheri Junction",
    )
    print("Inserted test record.")
    print(f"Total: {db.get_stats()['total_violations']}")
    for row in db.get_all():
        print(f"  {row['case_id']} | {row['plate_number']} | "
              f"{row['violation_types']} | {row['status']}")
