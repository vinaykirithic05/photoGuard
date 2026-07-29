from modules.metadata import MetadataVerifier
from modules.prnu import PRNUAnalyzer

image = "dataset/real/real1.jpg"

print("="*50)
print("        PHOTOGUARD AI DEMO")
print("="*50)

meta = MetadataVerifier()
meta.analyze(image)

prnu = PRNUAnalyzer()
prnu.analyze(image)

print("\n==============================")
print("Analysis Completed")
print("==============================")