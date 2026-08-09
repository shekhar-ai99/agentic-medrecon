# How to push this to your GitHub

I cannot push to your account (no access to your credentials). Here is how to
do it yourself. It takes about a minute.

## Option A — new repo via GitHub website + command line

1. On github.com, create a new EMPTY repository named `agentic-medrecon`
   (no README, no .gitignore, no license — this folder already has them).
   Make it public so the paper's link works for reviewers.

2. In this folder, run:

   ```bash
   git init
   git add .
   git commit -m "Initial commit: MARS multi-agent reconciliation scaffold"
   git branch -M main
   git remote add origin https://github.com/shekhar-ai99/agentic-medrecon.git
   git push -u origin main
   ```

   (Replace the URL if your username/repo name differs.)

## Option B — GitHub CLI (if you have `gh` installed)

```bash
git init && git add . && git commit -m "Initial commit: MARS scaffold"
gh repo create shekhar-ai99/agentic-medrecon --public --source=. --push
```

## After pushing

- Confirm the URL in the paper matches your actual repo URL. The paper cites:
  `https://github.com/shekhar-ai99/agentic-medrecon`
  If your repo lives elsewhere, update every occurrence in the .tex file
  (there are a few: abstract, contributions, reproducibility table, conclusion).

- Optional: mint a DOI by connecting the repo to Zenodo
  (https://zenodo.org) and cutting a release. Then replace
  "DOI to be assigned upon publication" in the paper with the real DOI.

## Verify it runs after cloning (sanity check for reviewers)

```bash
git clone https://github.com/shekhar-ai99/agentic-medrecon.git
cd agentic-medrecon
pip install -r requirements.txt
python -m pytest tests/ -q          # should show 7 passed
python scripts/demo.py --seed 7     # should print a full walkthrough
```
