"""Print unreviewed visual gates; never auto-approve images."""
from pathlib import Path
import json
B=Path(__file__).resolve().parents[1]
for p in sorted((B/'runs').glob('*/review_request_*.json')):
 if (p.parent/'result.json').exists():continue
 if not p.with_name(p.name.replace('review_request_','reviewed_')).exists():print(p, p.read_text())
