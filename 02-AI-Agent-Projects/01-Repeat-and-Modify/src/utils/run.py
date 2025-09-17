from langchain_core.runnables import Runnable
from typing import Dict

def stream(app:Runnable, inputs:Dict):
    for event in app.stream(inputs):
        for node, state in event.items():
            print("=============", f"[Node : '{node}']", "=============")
            for key, value in state.items():
                print(f"    {key} : \n      {value}\n")
        print("\n", "============"*15, "\n\n")