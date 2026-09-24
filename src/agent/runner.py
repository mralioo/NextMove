"""
CLI entry point for the U-Bahn operator agent.

Usage:
    # Interactive mode
    python src/agent/runner.py --data-dir "data/training dataset"
    
    # Evaluate all 9 training questions
    python src/agent/runner.py --data-dir "data/training dataset" --eval
    
    # Single question
    python src/agent/runner.py --data-dir "data/training dataset" \
        --question "U6 is suspended between Hallesches Tor and Kaiserin-Augusta-Str."
"""
import sys
import os
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.graph import build_agent_graph
from data_pipeline.loader import DataStore

# Minimal HumanMessage fallback if langchain isn't installed
try:
    from langchain_core.messages import HumanMessage
except ImportError:
    class HumanMessage:
        def __init__(self, content): self.content = content

TRAINING_QUESTIONS = [
    "There's a Guns N' Roses concert on June 23rd at the Uber Arena. What will the passenger flow look like at the neighboring stations and what measures should we take in operational sense?",
    "Give me an example of a passenger flow peak caused by bad weather in the week of July 20-26 and provide additional information like the time and station where the peak took place.",
    "Line U6 is suspended on a section between Hallesches Tor and Kaiserin-Augusta-Strasse stations. What is the reason behind this closure and how long will it last? How should the passengers be rerouted, which stations would become overloaded, and where should additional staff be deployed?",
    "At what time does the commute flow peak at Rudow station usually take place? Does it exceed the mean commute peak value across all stations?",
    "Which metro line has the worst energy-per-passenger efficiency ratio? What factors explain this inefficiency and what interventions would provide the largest improvement?",
    "Rank the five stations whose closure would fragment the network the most. For each station, estimate the number of passengers affected daily and suggest mitigation strategies.",
    "Identify three passenger-flow anomalies that cannot be explained by station closures on June 24th. Determine the most likely root causes using all of the available data.",
    "Are there stations whose passenger demand appears strongly dependent on another station despite no direct connection between them? Identify such pairs and explain the mechanism behind the dependency.",
    "During disruptions, which alternative routes do passengers actually prefer compared to the theoretically shortest routes? What does this reveal about passenger behavior?",
    "If you could invest in only one infrastructure improvement anywhere in the network, what should it be? Justify the recommendation using passenger flows, resilience, energy consumption, and historical disruption data.",
    "During InnoTrans we expect a major surge in passenger flow around Messe Berlin towards the city center. Can you suggest an unconventional alternative route not based on the shortest path?",
]

def initialize_agent(data_dir: str):
    """Initializes DataStore and returns compiled agent graph."""
    try:
        t0 = time.time()
        DataStore.initialize(data_dir)
        t1 = time.time()
        print(f"DataStore initialization took {t1 - t0:.2f} seconds.")
    except Exception as e:
        print(f"Error initializing DataStore: {e}")
        print(f"Make sure the data_dir '{data_dir}' contains the necessary CSV files.")
        sys.exit(1)
        
    return build_agent_graph()

import time

def ask(agent, question: str) -> str:
    """Invoke the agent with a question and return its response."""
    state = {"messages": [HumanMessage(content=question)]}
    t0 = time.time()
    result = agent.invoke(state)
    t1 = time.time()
    latency = t1 - t0
    
    resp = result.get('final_response', "Error: No response generated.")
    return f"{resp}\n\n[Agent Execution Latency: {latency:.2f} seconds]"

def run_eval(agent, output_file: str = 'evaluation/eval_answers.txt'):
    """Run agent against all training questions and save output."""
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for i, q in enumerate(TRAINING_QUESTIONS):
            print(f"\nEvaluating Q{i+1}/{len(TRAINING_QUESTIONS)}: {q}")
            f.write(f"Question {i+1}:\n{q}\n\n")
            try:
                ans = ask(agent, q)
                print("Response received.")
                f.write(f"Answer:\n{ans}\n")
            except Exception as e:
                print(f"Error evaluating question {i+1}: {e}")
                f.write(f"Answer:\nError: {e}\n")
            f.write("-" * 80 + "\n\n")
            
    print(f"\nEvaluation complete. Results saved to {output_file}")

def run_interactive(agent):
    """Run an interactive REPL loop."""
    print("\n" + "="*50)
    print("U-Bahn Operations Intelligence Agent (Vanilla / Azure Responses API)")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")
    
    while True:
        try:
            q = input("Operator > ")
            if q.strip().lower() in ['exit', 'quit']:
                break
            if not q.strip():
                continue
                
            ans = ask(agent, q)
            print("\nAgent:")
            print(ans)
            print("\n" + "-"*50 + "\n")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}\n")

def main():
    parser = argparse.ArgumentParser(description="U-Bahn Operator Intelligence Agent Runner")
    parser.add_argument("--data-dir", required=True, help="Path to directory containing CSV data")
    parser.add_argument("--eval", action="store_true", help="Run evaluation on all training questions")
    parser.add_argument("--question", type=str, help="Single question to ask the agent")
    parser.add_argument("--output", type=str, default="evaluation/eval_answers.txt", help="Output file for evaluation")
    
    args = parser.parse_args()
    
    print("Initializing agent and loading data...")
    agent = initialize_agent(args.data_dir)
    print("Initialization complete.")
    
    if args.eval:
        run_eval(agent, args.output)
    elif args.question:
        ans = ask(agent, args.question)
        print(f"\nQuestion: {args.question}")
        print("\nAgent Response:")
        print(ans)
    else:
        run_interactive(agent)

if __name__ == '__main__':
    main()
