"""Évaluation déterministe d'une politique SAC et export CSV/JSON par essai."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC
from src.assembly_env import TenonMortaiseEnv

def find_model(run: Path) -> Path:
 """Prefer the completed policy; otherwise evaluate the latest checkpoint."""
 final_model = run / "model.zip"
 if final_model.is_file(): return final_model
 checkpoints = sorted((run / "checkpoints").glob("*.zip"), key=lambda path: path.stat().st_mtime)
 if checkpoints:
  checkpoint = checkpoints[-1]
  print(f"Modèle final absent; évaluation du checkpoint: {checkpoint.name}")
  return checkpoint
 raise FileNotFoundError(
  f"Aucun modèle à évaluer dans {run}. Attendu: model.zip ou checkpoints/*.zip. "
  "Terminez ou relancez l'entraînement."
 )

def main():
 p=argparse.ArgumentParser(); p.add_argument("--run",type=Path,required=True); p.add_argument("--episodes",type=int,default=20); p.add_argument("--render",action="store_true"); p.add_argument("--render-speed",type=float,default=1.0,help="1=temps réel, 0.25=quatre fois plus lent"); args=p.parse_args()
 env=TenonMortaiseEnv(args.run/"config.yaml", "human" if args.render else None, args.render_speed); model=SAC.load(find_model(args.run),env=env)
 rows=[]
 for episode in range(args.episodes):
  obs,_=env.reset(seed=100+episode); done=False; reward=0.; step=0; max_force=max_torque=0.
  while not done:
   action,_=model.predict(obs,deterministic=True); obs,r,terminated,truncated,info=env.step(action); reward+=r; step+=1; done=terminated or truncated; max_force=max(max_force,info["force"]); max_torque=max(max_torque,info["torque"])
  rows.append({"episode":episode,"success":info["success"],"steps":step,"reward":reward,"final_position_error":info["position_error"],"final_rotation_error":info["rotation_error"],"max_force":max_force,"max_torque":max_torque,"unsafe":info["unsafe"]})
 with (args.run/"evaluation.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
 summary={"success_rate":float(np.mean([r["success"] for r in rows])),"mean_steps":float(np.mean([r["steps"] for r in rows])),"effort_terminations":int(sum(r["unsafe"] for r in rows))}
 (args.run/"evaluation.json").write_text(json.dumps(summary,indent=2)); env.close(); print(summary)
if __name__=="__main__": main()
