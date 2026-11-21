import sqlite3
from pathlib import Path

MEETING_ID = "demo1"
DB_PATH = Path(__file__).parent.parent / "data" / "meetings.db"

LINES = [
    ("Ravi", "Good morning everyone. Let's start the weekly sync for the Campus Connect app."),
    ("Priya", "Good morning Ravi. I finished the login screen and connected it to Firebase yesterday."),
    ("Ravi", "Great. Did you test it on Android and iPhone?"),
    ("Priya", "I tested on Android only. I still need to check the iPhone version."),
    ("Anil", "I can test the iPhone build on Thursday. I have a device from the college lab in Hyderabad."),
    ("Ravi", "Perfect. Anil, please send the test report to Priya by Friday."),
    ("Anil", "Sure, I will send it by Friday evening."),
    ("Priya", "There is one problem with the payment page. The Razorpay test key is not working."),
    ("Ravi", "I will contact Razorpay support today and ask for a new key."),
    ("Priya", "Thanks. Without it I cannot finish the checkout flow."),
    ("Anil", "Should we also plan a demo for Dr. Sharma?"),
    ("Ravi", "Yes. Dr. Sharma is visiting from Chennai on October 28th, so we must be ready before that."),
    ("Priya", "I will prepare the demo slides by Monday."),
    ("Ravi", "Good. Anil, can you also write the test cases for the chat feature?"),
    ("Anil", "Yes, I will write them next week and share them in the team folder."),
    ("Priya", "The chat feature is slower on weak networks. We need to fix the image upload."),
    ("Ravi", "Let's make that the first priority after the payment page."),
    ("Priya", "I think compressing the images before upload will help."),
    ("Anil", "I agree. I can measure the upload time before and after the change."),
    ("Ravi", "Great idea. Priya, please add the compression by Wednesday."),
    ("Priya", "Okay, I will do it on Wednesday and tell you when it is ready."),
    ("Ravi", "One more thing. The college will give us a server on Microsoft Azure next month."),
    ("Anil", "Will we move the database there?"),
    ("Ravi", "Yes, but only after the demo. I will talk to the IT office about it."),
    ("Priya", "Then I will keep Firebase until the demo is finished."),
    ("Ravi", "Exactly. Does anyone have other topics?"),
    ("Anil", "No, that is all from my side."),
    ("Ravi", "Thank you both. Let's meet again next Monday at 10 AM."),
]

db = sqlite3.connect(DB_PATH)
db.execute("""CREATE TABLE IF NOT EXISTS segments(
    meeting_id TEXT, speaker TEXT, start REAL, end REAL, text TEXT)""")
db.execute("DELETE FROM segments WHERE meeting_id=?", (MEETING_ID,))

t = 0.0
for speaker, text in LINES:
    duration = max(1.5, len(text.split()) * 0.35)
    db.execute("INSERT INTO segments VALUES (?,?,?,?,?)",
               (MEETING_ID, speaker, round(t, 2), round(t + duration, 2), text))
    t += duration + 0.6

db.commit()
db.close()
print(f"Saved {len(LINES)} lines as meeting '{MEETING_ID}' ({t:.0f} s long)")