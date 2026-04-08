from src.DataCleaning import readDf, cleanDf

try:
    filename = "data/Book1.csv"
    df = readDf()
    cols = ["internet_access_hours", "social_media_hours", "attention_span_minutes", "productivity_score", "academic_motivation"]
    cleaned = df
    for col in cols:
        outliers, msg = cleanDf(cleaned, col=col, debug=False)
        print(f"- Cleaning column: {col} Using {msg}")

        if len(outliers) > 0:
            cleaned = cleaned.drop(index=outliers)
    print(f"Remaining rows: {len(cleaned)}")
except Exception as e:
    print(f"Error occurs: {e}")
