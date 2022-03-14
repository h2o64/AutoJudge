# Compress the data
import zipfile

with zipfile.ZipFile('data/public.zip', 'r') as zip_ref:
    zip_ref.extractall('data/')