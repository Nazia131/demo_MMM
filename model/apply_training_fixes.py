"""Patch model/train.py and model/gat_model.py in place to fix the collapsed training.

Run from the project root:  python apply_training_fixes.py
Makes *.bak backups first. Aborts without changing anything if a target line
is not found (e.g. you already edited it) and tells you which one.
"""
import os, re, shutil, sys

TRAIN, GAT = os.path.join("model", "train.py"), os.path.join("model", "gat_model.py")
for f in (TRAIN, GAT):
    if not os.path.isfile(f):
        sys.exit(f"{f} not found - run this from the project root (the folder containing model/).")

t, g = open(TRAIN, encoding="utf-8").read(), open(GAT, encoding="utf-8").read()
problems = []

def sub(text, pattern, repl, label, flags=0):
    new, n = re.subn(pattern, repl, text, count=1, flags=flags)
    if n != 1:
        problems.append(label)
    return new

# 1. lower learning rate
t = sub(t, r"^_LEARNING_RATE\s*=\s*[0-9.eE-]+", "_LEARNING_RATE = 0.001", "learning rate line", re.M)

# 2. class-weighted training loss (the eval loss stays unweighted)
t = sub(
    t,
    r"(?P<i>[ \t]*)criterion = nn\.NLLLoss\(\)\n(?P<gap>\s*)os\.makedirs\(_CHECKPOINT_DIR",
    lambda m: (
        f"{m['i']}_ys = torch.tensor([int(d.y.view(-1)[0]) for d in splits['train']])\n"
        f"{m['i']}_pos = max(int((_ys == 1).sum()), 1)\n"
        f"{m['i']}_neg = max(int((_ys == 0).sum()), 1)\n"
        f"{m['i']}_w = torch.tensor([len(_ys) / (2.0 * _neg), len(_ys) / (2.0 * _pos)], dtype=torch.float)\n"
        f"{m['i']}criterion = nn.NLLLoss(weight=_w.to(device))\n"
        f"{m['gap']}os.makedirs(_CHECKPOINT_DIR"
    ),
    "training loss line (criterion = nn.NLLLoss() before os.makedirs)",
)

# 3. choose the best checkpoint by validation AUC (F1 rewards 'always harmful')
t = sub(t, r"if val_f1 > best_val_f1:\n(\s*)best_val_f1 = val_f1",
        r'if val_select > best_val_f1:\n\1best_val_f1 = val_select', "checkpoint selection (if val_f1 > best_val_f1)")
t = sub(t, r'(val_f1 = val_metrics\.get\("f1", 0\.0\)\n)',
        r'\1        _auc = val_metrics.get("auc")\n        val_select = _auc if (_auc is not None and _auc == _auc) else val_f1  # AUC, not F1\n', "val_f1 assignment")

# 4. lower dropout in the GAT
g = sub(g, r"^GAT_DROPOUT:\s*float\s*=\s*0\.6", "GAT_DROPOUT: float = 0.3", "GAT_DROPOUT line", re.M)

if problems:
    sys.exit("Nothing was changed. Could not find: " + "; ".join(problems) +
             "\nSend me those lines of your train.py / gat_model.py and I will adjust the patch.")

for f, s in ((TRAIN, t), (GAT, g)):
    shutil.copy2(f, f + ".bak")
    open(f, "w", encoding="utf-8").write(s)
print("Patched model/train.py and model/gat_model.py (backups: *.bak).")
print("Changes: lr 0.005->0.001, class-weighted loss, best checkpoint by val AUC, GAT dropout 0.6->0.3.")