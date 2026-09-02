
import json
from pathlib import Path

audit = json.loads(Path(".state/remote_storage_audit.json").read_text())

print("=" * 80)
print("AUDITORÍA DE ALMACENAMIENTO Y VERSIONES VIEJAS EN EL CLUSTER REMOTO")
print("=" * 80)

total_cluster_exp_mb = 0

for node, data in audit.items():
    print("\n" + "-" * 70)
    print("🖥️  NODO: " + node.upper())
    if "error" in data:
        print("   Error: " + str(data["error"]))
        continue
    
    print("   Espacio en Disco (Root): " + str(data.get("df_root", "N/A")))
    print("   Tamaño total ~/Documents/Biohub: " + str(data.get("biohub_total_size", "N/A")))
    print("   Dataset de entrenamiento (~/data): " + str(data.get("data_size", "N/A")))
    
    exps = data.get("experiments", [])
    if exps:
        print("\n   📁 Desglose de Experimentos en este nodo (" + str(len(exps)) + "):")
        for e in exps:
            name = e.get("name")
            size = e.get("total_size")
            cp_count = e.get("checkpoint_count", 0)
            cp_mb = e.get("checkpoint_mb", 0)
            total_cluster_exp_mb += cp_mb
            
            # Categorize
            tag = ""
            if "EXP-051" in name:
                tag = " [🏆 ACTUAL CAMPEÓN KAGGLE]"
            elif "EXP-052" in name or "EXP-050" in name:
                tag = " [🌟 DEEPGRAPH VIGENTE]"
            elif "EXP-053" in name or "EXP-054" in name or "EXP-055" in name:
                tag = " [🚀 NUEVA VARIANTE]"
            else:
                tag = " [📦 VERSIÓN ANTIGUA / LEGACY]"
                
            print("     • " + name + tag)
            print("       - Tamaño total carpeta: " + str(size))
            print("       - Checkpoints .pth: " + str(cp_count) + " (" + str(cp_mb) + " MB)")
    else:
        print("   📁 Sin experimentos de Biohub en disco.")
        
    other_docs = data.get("other_docs", [])
    if other_docs:
        print("\n   📂 Otras carpetas en ~/Documents:")
        for od in other_docs:
            print("     - " + od["name"] + ": " + od["size"])
