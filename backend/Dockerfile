# Use Miniforge3 for Mamba/Conda support
FROM condaforge/miniforge3:latest

WORKDIR /app

# Install Conda dependencies from conda-forge
# python=3.10 is specified to ensure compatibility
# tblite-python and rdkit are main requirements
RUN mamba install -y -c conda-forge python=3.10 tblite tblite-python rdkit numpy

# Install Pip dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]