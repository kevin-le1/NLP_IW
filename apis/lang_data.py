from datasets import load_dataset

dset = datasets.load_dataset("SEACrowd/cc100", trust_remote_code=True)

# Load the dataset using the default config
dset = sc.load_dataset("cc100", schema="seacrowd")
# Check all available subsets (config names) of the dataset
print(sc.available_config_names("cc100"))