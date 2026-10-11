import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_architecture_diagram(output_path="docs/system_architecture.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Colors
    c_raw = "#34495e"
    c_prep = "#2980b9"
    c_model = "#8e44ad"
    c_xai = "#d35400"
    c_bi = "#27ae60"
    c_deploy = "#c0392b"

    def draw_box(x, y, w, h, title, subtitle, color):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15", ec="none", fc=color, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.62, title, ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        ax.text(x + w/2, y + h*0.28, subtitle, ha="center", va="center", color="#ecf0f1", fontsize=8.5)

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2.0, mutation_scale=15))

    # Title
    ax.text(7, 7.6, "Explainable AML Pipeline & Business Risk Architecture", ha="center", va="center", fontsize=15, fontweight="bold", color="#1a252f")

    # Row 1: Pipeline Pipeline Stages
    # Box 1: Raw Data
    draw_box(0.5, 4.8, 2.4, 1.8, "Raw Dataset", "SAML-D (9.5M rows)\n12 Features, 0.1% Pos", c_raw)
    draw_arrow(2.9, 5.7, 3.5, 5.7)

    # Box 2: Preprocessing
    draw_box(3.5, 4.8, 2.8, 1.8, "Zero-Leakage Prep", "Causal History (shift=1)\nChronological Split (70/15/15)", c_prep)
    draw_arrow(6.3, 5.7, 6.9, 5.7)

    # Box 3: Ensemble ML
    draw_box(6.9, 4.8, 3.2, 1.8, "Ensemble ML Engine", "LightGBM + Random Forest\nIsolation Forest + LogReg\nRank Norm & Isotonic Cal", c_model)
    draw_arrow(10.1, 5.7, 10.7, 5.7)

    # Box 4: Explainability
    draw_box(10.7, 4.8, 2.8, 1.8, "SHAP Explainability", "TreeExplainer Attribution\nBeeswarm + Waterfalls\nTop-3 Plain-English Reasons", c_xai)

    # Row 2: Downstream Business Intelligence & Deployment
    draw_arrow(12.1, 4.8, 12.1, 3.5)
    draw_arrow(12.1, 3.5, 10.7, 2.3)

    # Box 5: BI Risk Layer
    draw_box(6.9, 1.4, 3.5, 1.8, "BI Decision Layer", "Expected Loss = P_cal x Amt\nRisk Tiers (Critical/High/Med/Low)\nCost-Optimal Threshold ($45M Saved)", c_bi)
    draw_arrow(6.9, 2.3, 4.2, 2.3)

    # Box 6: Deployment & Interfaces
    draw_box(0.8, 1.4, 3.4, 1.8, "Application Layer", "FastAPI Backend (Port 8000)\nStreamlit Executive UI (Port 7860)\nDocker & Hugging Face Spaces", c_deploy)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Architecture diagram saved to {output_path}")

if __name__ == "__main__":
    generate_architecture_diagram()
