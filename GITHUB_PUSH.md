# GitHub Repository Synchronization Guide

This document contains the exact commands to push the complete repository to your remote GitHub repository without uploading the raw ~1 GB `SAML-D.csv` dataset (which is safely excluded via `.gitignore`).

---

### Step 1: Initialize Git Repository (if not already initialized)
```bash
git init
```

### Step 2: Stage All Clean Code, Models, Outputs, and Documentation
```bash
# Add all tracked files (.gitignore automatically excludes SAML-D.csv, parquets, pycache)
git add .

# Verify status: confirm that SAML-D.csv and raw parquet files are NOT staged
git status
```

### Step 3: Create Initial Structured Commit
```bash
git commit -m "feat: complete explainable AML fraud detection & business risk pipeline"
```

### Step 4: Configure Remote Origin & Main Branch
Replace `<YOUR_GITHUB_REPOSITORY_URL>` with your actual GitHub repository URL (e.g., `https://github.com/your-username/aml-fraud-detection.git`):
```bash
# Set primary branch to main
git branch -M main

# Add remote repository
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
```

### Step 5: Push to Remote Repository
```bash
git push -u origin main
```

---

### Note on Large Model Artifacts
If your remote repository enforces a strict 50 MB or 100 MB file limit on `models/ensemble.joblib`:
```bash
# Optional: track large models via Git LFS
git lfs install
git lfs track "models/*.joblib"
git add .gitattributes
git commit -m "chore: configure git-lfs for ensemble model artifact"
git push origin main
```
