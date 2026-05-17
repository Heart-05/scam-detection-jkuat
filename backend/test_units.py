from app import preprocess_text, generate_explanation, get_risk_level

# Test preprocess_text
print("=== preprocess_text tests ===")
print(preprocess_text("Hello WORLD!"))
print(preprocess_text("Visit http://scam.com now!!!"))
print(preprocess_text("   extra   spaces   "))

# Test generate_explanation
print("\n=== generate_explanation tests ===")
print(generate_explanation("Scam", ["helb", "urgent", "verify"]))
print(generate_explanation("Scam", []))
print(generate_explanation("Legitimate", []))

# Test get_risk_level
print("\n=== get_risk_level tests ===")
print(get_risk_level("Scam", 90))
print(get_risk_level("Scam", 70))
print(get_risk_level("Scam", 50))
print(get_risk_level("Legitimate", 86))