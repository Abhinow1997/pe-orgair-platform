
# Git Basic Workflow

## Overview
Git is a distributed version control system that tracks changes to your code.

## Basic Commands

### 1. Initialize a Repository
```bash
git init
```

### 2. Clone a Repository
```bash
git clone <repository-url>
```

### 3. Check Status
```bash
git status
```

### 4. Stage Changes
```bash
git add <file>
git add .  # Stage all changes
```

### 5. Commit Changes
```bash
git commit -m "Descriptive message"
```

### 6. View History
```bash
git log
```

### 7. Create a Branch
```bash
git branch <branch-name>
git checkout <branch-name>
```

### 8. Push to Remote
```bash
git push origin <branch-name>
```

### 9. Pull from Remote
```bash
git pull origin <branch-name>
```

### 10. Merge Branches
```bash
git checkout main
git merge <branch-name>
```

## Typical Workflow
1. Create a new branch for your feature
2. Make changes and stage them
3. Commit with clear messages
4. Push to remote repository
5. Create a pull request
6. Merge after review
