"""Independent preparation menu; never downloads new research candidates."""
from lithiumscope.cli.display import clear_screen
from lithiumscope.cli.prompts import choose


def run():
    from lithiumscope.core.logging.session import run_log
    from lithiumscope.datasets.management.diagnostic_provisioning import prepare_diagnostic_source
    from lithiumscope.experiments.data_lab.workflow import prepare
    from lithiumscope.cli.eda_verification import write_verification

    print("PREPARACIÓN Y TRATAMIENTO DE DATOS\n1. Modelo 1\n2. Modelo 2\n3. Ambos modelos\n0. Volver")
    choice=choose("Seleccione una opción [0-3]: ",set("0123"))
    if choice=="0":
        return False
    clear_screen()
    groups=("model_1","model_2") if choice=="3" else (f"model_{choice}",)
    records={}
    for group in groups:
        try:
            with run_log("eda",group+"_preparation"):
                state,record=prepare(group,prepare_diagnostic_source(group))
            records[group]={"state":state,**record}
            print(f"[{state.upper()}] {group}")
        except Exception as exc:
            records[group]={"state":"blocked","reason":str(exc)}
            print(f"[BLOQUEADO] {group}: {exc}")
    state="COMPLETADO" if all(r["state"]=="complete" for r in records.values()) else "PARCIAL/BLOQUEADO"
    print(f"Informe de preparación: {write_verification(state,records,scope='train_only_data_preparation')}")
    # Return to lab only after user can inspect errors/results.
    return True
