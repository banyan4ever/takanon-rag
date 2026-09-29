import sys

import db

TEST_NUMBER = "__test__"

try:
    conn = db.connect()
except Exception as e:
    sys.exit(f"❌ החיבור למסד הנתונים נכשל: {e}")

with conn:
    print("✅ התחברנו בהצלחה למסד הנתונים ב-Neon")
    print(f"מספר הסעיפים בהתחלה: {db.count_sections(conn)}")

    db.upsert_section(conn, TEST_NUMBER, "פרק בדיקה", "זהו סעיף בדיקה זמני")
    updated = db.upsert_section(conn, TEST_NUMBER, "פרק בדיקה", "זהו סעיף בדיקה זמני (עודכן)")
    print(f"➕ נוסף ועודכן סעיף בדיקה (גרסה {updated['version']}). מספר הסעיפים: {db.count_sections(conn)}")

    rows = db.get_sections(conn, [TEST_NUMBER])
    assert len(rows) == 1 and rows[0]["version"] == 2, "סעיף הבדיקה לא נקרא כמצופה"
    r = rows[0]
    print(f"📖 נקרא בחזרה: סעיף {r['number']} | {r['chapter']} | {r['text']} | גרסה {r['version']} | עודכן {r['updated_at']:%Y-%m-%d %H:%M}")

    db.delete_section(conn, TEST_NUMBER)
    assert not db.get_sections(conn, [TEST_NUMBER]), "סעיף הבדיקה לא נמחק"
    print(f"🗑️ סעיף הבדיקה נמחק. מספר הסעיפים: {db.count_sections(conn)}")

print("✅ הבדיקה עברה בהצלחה")
