import sys
import time

import config
import vectors

TEST_ID = "__test__"

try:
    pc = vectors.connect()
    existed = pc.has_index(config.PINECONE_INDEX)
    index = vectors.get_index(pc)
except Exception as e:
    sys.exit(f"❌ החיבור ל-Pinecone נכשל: {e}")

print(f"✅ התחברנו ל-Pinecone. האינדקס '{config.PINECONE_INDEX}' {'כבר קיים' if existed else 'נוצר עכשיו'} (ממד {vectors.DIMENSION})")
print(f"מספר הווקטורים בהתחלה: {vectors.count_vectors(index)}")

dummy = [0.1] * vectors.DIMENSION
vectors.upsert_vectors(index, [(TEST_ID, dummy, {"chapter": "פרק בדיקה", "text": "וקטור בדיקה"})])
print("➕ נשמר וקטור בדיקה")

# Pinecone is eventually consistent: a fresh write can take a few seconds to show up
for _ in range(20):
    found = vectors.fetch_vectors(index, [TEST_ID])
    if TEST_ID in found:
        break
    time.sleep(1)
else:
    sys.exit("❌ וקטור הבדיקה לא נמצא אחרי 20 שניות")

v = found[TEST_ID]
assert len(v.values) == vectors.DIMENSION, "ממד הווקטור שנקרא שגוי"
print(f"📖 נקרא בחזרה: מזהה {v.id} | ממד {len(v.values)} | {v.metadata}")
print(f"מספר הווקטורים עכשיו: {vectors.count_vectors(index)}")
print("✅ הבדיקה עברה בהצלחה")
