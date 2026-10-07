from pathlib import Path
import subprocess,sys
B=Path(__file__).resolve().parents[1];r=Path(sys.argv[1])
with (r/'validation.log').open('w') as f:
 for n in ['validate_capture.py','validate_brainco.py','validate_contacts.py']:
  subprocess.run([sys.executable,str(B/'scripts'/n),'--run',str(r)],stdout=f,stderr=subprocess.STDOUT,check=True)
 subprocess.run([sys.executable,str(B/'scripts/report_run.py'),str(r)],stdout=f,stderr=subprocess.STDOUT,check=True)
print('VALIDATED',r)
