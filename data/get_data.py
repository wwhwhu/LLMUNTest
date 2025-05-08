from datasets import load_dataset

wikitext = load_dataset("wikitext", "wikitext-103-v1")
# Save the dataset to a local directory
wikitext.save_to_disk("data/wikitext-103-v1")

wmdp_bio = load_dataset("cais/wmdp", name="wmdp-bio")
# Save the dataset to a local directory
wmdp_bio.save_to_disk("data/wmdp-bio")