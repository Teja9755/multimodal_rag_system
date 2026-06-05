from sqlalchemy import create_engine, text

# Make sure this matches your finance_multimodal_rag_db
DATABASE_URL = "postgresql+psycopg2://postgres:Pass%40123@localhost:5433/finance_multimodal_rag_db"

engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    rows = conn.execute(
        text("""
            SELECT id, document_name, source_page, chunk_type, content
            FROM document_chunks
            ORDER BY document_name, source_page
        """)
    ).fetchall()

with open("stored_chunks.txt", "w", encoding="utf-8") as f:
    for row in rows:
        f.write("\n")
        f.write("=" * 120 + "\n")
        f.write(f"ID: {row.id}\n")
        f.write(f"DOCUMENT: {row.document_name}\n")
        f.write(f"PAGE: {row.source_page}\n")
        f.write(f"TYPE: {row.chunk_type}\n")
        f.write("-" * 120 + "\n")
        f.write(row.content + "\n")

print("stored_chunks.txt created successfully")