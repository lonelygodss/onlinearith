import json
import matplotlib.pyplot as plt
import numpy as np
import os

# ================= Configuration =================
JSON_FILE = "qwen3_benchmark_results.json"  # Save your JSON data to this file
MODEL_NAME = "Qwen3-Benchmark"
# ===============================================

def load_data():
    """Loads the specific JSON structure provided by the user."""
    # If file exists, load it. If not, create it with the provided data for demonstration.
    if not os.path.exists(JSON_FILE):
        print(f"File {JSON_FILE} not found. Please save your JSON data to this file.")
        return None

    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_metrics(data):
    """
    Parses the JSON to extract:
    1. Overall Scores (MMLU, GSM8K)
    2. MMLU Categories (STEM, Humanities, etc.)
    3. Specific MMLU Subjects (Anatomy, Logic, etc.)
    """
    
    # Containers
    overall = {}
    mmlu_cats = {}
    mmlu_subjects = {}

    # Define MMLU Broad Categories for filtering
    broad_cats = ['mmlu_stem', 'mmlu_humanities', 'mmlu_social_sciences', 'mmlu_other']

    for task, metrics in data.items():
        # 1. Get the score (handle different keys)
        score = 0.0
        if "acc,none" in metrics:
            score = metrics["acc,none"]
        elif "exact_match,strict-match" in metrics:
            score = metrics["exact_match,strict-match"]
        
        # Convert to percentage
        score = round(score * 100, 2)

        # 2. Categorize
        if task == 'gsm8k':
            overall['GSM8K'] = score
        elif task == 'mmlu':
            overall['MMLU (Avg)'] = score
        elif task in broad_cats:
            # Clean name: "mmlu_stem" -> "STEM"
            clean_name = task.replace("mmlu_", "").replace("_", " ").title()
            if clean_name == "Stem": clean_name = "STEM"
            mmlu_cats[clean_name] = score
        elif task.startswith('mmlu_'):
            # These are specific subjects (e.g., mmlu_anatomy)
            clean_name = task.replace("mmlu_", "").replace("_", " ").title()
            mmlu_subjects[clean_name] = score

    return overall, mmlu_cats, mmlu_subjects

def plot_radar_mmlu(cats, title):
    """Plots a Radar chart for MMLU Categories (STEM, Humanities, etc.)"""
    labels = list(cats.keys())
    values = list(cats.values())
    
    # Close the loop
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw axes
    plt.xticks(angles[:-1], labels, size=12)
    
    # Draw y-labels
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="grey", size=10)
    plt.ylim(0, 100)

    # Plot
    ax.plot(angles, values, linewidth=2, linestyle='solid', label="MMLU Categories", color='#d62728')
    ax.fill(angles, values, '#d62728', alpha=0.25)

    plt.title(f"{title} - Knowledge Breakdown", size=16, weight='bold', y=1.08)
    
    # Add values
    for angle, value in zip(angles[:-1], values[:-1]):
        ax.text(angle, value + 10, f"{value}", ha='center', va='center', fontsize=11, weight='bold')

    plt.tight_layout()
    plt.savefig("chart_mmlu_radar.png", dpi=300)
    print("Saved: chart_mmlu_radar.png")
    plt.close()

def plot_top_bottom_subjects(subjects):
    """Plots the Top 5 and Bottom 5 specific subjects."""
    # Sort subjects by score
    sorted_items = sorted(subjects.items(), key=lambda x: x[1])
    
    # Get Top 5 and Bottom 5
    bottom_5 = sorted_items[:5]
    top_5 = sorted_items[-5:]
    
    # Combine for plotting
    plot_data = bottom_5 + top_5
    labels = [x[0] for x in plot_data]
    values = [x[1] for x in plot_data]
    
    # Colors: Red for low, Green for high
    colors = ['#ff9999']*5 + ['#99ff99']*5

    plt.figure(figsize=(12, 8))
    bars = plt.barh(labels, values, color=colors)
    
    plt.xlim(0, 100)
    plt.xlabel("Accuracy (%)")
    plt.title("MMLU: Weakest (Red) vs Strongest (Green) Subjects", fontsize=14, weight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, 
                 f'{width}%', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig("chart_mmlu_subjects.png", dpi=300)
    print("Saved: chart_mmlu_subjects.png")
    plt.close()

def plot_overview_bar(overall):
    """Plots GSM8K vs MMLU Total."""
    labels = list(overall.keys())
    values = list(overall.values())
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=['#1f77b4', '#ff7f0e'], width=0.5)
    
    plt.ylim(0, 100)
    plt.ylabel("Score (%)")
    plt.title("Overall Benchmark Summary", fontsize=14, weight='bold')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                 f'{height}', ha='center', va='bottom', fontsize=12, weight='bold')
        
    plt.tight_layout()
    plt.savefig("chart_overview.png", dpi=300)
    print("Saved: chart_overview.png")
    plt.close()

if __name__ == "__main__":
    data = load_data()
    
    if data:
        print("Data loaded successfully.")
        overall, mmlu_cats, mmlu_subjects = extract_metrics(data)
        
        print("-" * 30)
        print(f"Overall: {overall}")
        print(f"Categories: {mmlu_cats}")
        print("-" * 30)
        
        # Generate the 3 charts
        plot_overview_bar(overall)
        if mmlu_cats:
            plot_radar_mmlu(mmlu_cats, MODEL_NAME)
        if mmlu_subjects:
            plot_top_bottom_subjects(mmlu_subjects)
            
        print("\nVisualization Complete. Please check the 3 generated .png files.")
