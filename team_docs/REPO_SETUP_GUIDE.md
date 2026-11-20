# How to set up the two GitHub repositories

You have two folders inside "Hackathon project":

| Folder | Becomes repo | Who sees it | What is in it |
|---|---|---|---|
| `main_project/` | **ridefare** (your existing repo) | Judges / public | Only what is presented: notebook, app, model, data, final PPT, README, requirements |
| `team_docs/` + everything else | **ridefare-team** (new, PRIVATE) | Your team | Everything: all PPTs, study guide, instructions PDF, learning notes, practice files, old code |

---

## Repo 1: `ridefare` (presentation / main_project)

Your existing repo already has old files. We replace them with the clean `main_project` folder.

```bash
# 1. Go to the main_project folder
cd "/c/Users/tmanghnani/OneDrive - DXC Production/Desktop/Hackathon project/main_project"

# 2. Start Git here and link it to your existing repo
git init
git branch -M main
git remote add origin https://github.com/manghnanitany1998-del/ridefare.git

# 3. Add everything and check the list
git add .
git status          # should show: README.md, requirements.txt, notebooks/, app/, data/, models/, presentation/, reports/

# 4. Save and upload (replaces the old repo content with this clean version)
git commit -m "Final main_project: notebook, model, app and presentation"
git push -u origin main --force
```

`--force` is used **only this once**, to replace the old content. From now on use a normal `git push`.

After you run the notebook (which creates `models/uber_fare_model.pkl` and `reports/`), upload again:
```bash
git add .
git commit -m "Add trained model and results"
git push
```

---

## Repo 2: `ridefare-team` (everything, private)

**Step 1: create it on GitHub**
1. Go to github.com/new
2. Name: `ridefare-team`
3. Choose **Private**
4. Leave all boxes unticked, click **Create repository**

**Step 2: upload the whole Hackathon project folder**
```bash
cd "/c/Users/tmanghnani/OneDrive - DXC Production/Desktop/Hackathon project"

git init
git branch -M main
git remote add origin https://github.com/manghnanitany1998-del/ridefare-team.git

git add .
git status          # check venv/ is NOT in the list
git commit -m "Full RideFare project: code, docs, presentations and notes"
git push -u origin main
```

Run the **team repo steps first**, then the presentation repo steps. That way the team repo gets a full copy of `main_project/` as normal files, and the presentation repo is created inside `main_project/` afterwards (its `.git` folder is ignored by the team repo).

**Step 3: give your team access**
GitHub → `ridefare-team` → **Settings → Collaborators → Add people** → enter each teammate's GitHub username.

---

## If something goes wrong
| Problem | Fix |
|---|---|
| `remote origin already exists` | `git remote set-url origin <url>` |
| Push asks for login | A browser window opens; sign in to GitHub |
| Network / proxy error | Switch to phone hotspot and run the push again |
| File too large (> 100 MB) | Check `git status`; venv must not be listed |
| Wrong email on commits | `git config --global user.email "manghnanitany1998@gmail.com"` |
