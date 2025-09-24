import streamlit as st
import numpy as np

from tree_of_thought_perception import perception_tree_of_thought, Node, l2_normalize

st.set_page_config(page_title="SPRNS UI", layout="wide")
st.title("SPRNS Tree-of-Thought Perception UI")

# -------------------- User Inputs --------------------
obs_input = st.text_area("Enter observation vector (comma-separated numbers):", "0.9,0.1,0.2")
beam_size = st.number_input("Beam size:", min_value=1, max_value=10, value=3)
depth = st.number_input("Depth:", min_value=1, max_value=5, value=3)
lambda_likelihood = st.number_input("Lambda (likelihood scaling):", min_value=0.1, max_value=50.0, value=12.0)
beta_attention = st.number_input("Beta (attention softmax):", min_value=0.1, max_value=10.0, value=5.0)

# -------------------- Parse Observation --------------------
try:
    obs_vec = np.array([float(x.strip()) for x in obs_input.split(',')])
except Exception as e:
    st.error(f"Invalid observation vector: {e}")
    st.stop()

# -------------------- Default initial hypotheses --------------------
initial_hypotheses = [
    {'rep': np.array([1.0, 0.0, 0.0]), 'prior': 0.6, 'label': 'A'},
    {'rep': np.array([0.0, 1.0, 0.0]), 'prior': 0.4, 'label': 'B'}
]

# -------------------- Custom Expansion Function --------------------
def custom_expand_fn(node: Node):
    # small perturbations towards prototypes
    return [
        {'rep': node.rep * 0.9 + 0.1 * np.array([1.0, 0.0, 0.0]), 'prior': 0.3, 'label': 'towards_A'},
        {'rep': node.rep * 0.9 + 0.1 * np.array([0.0, 1.0, 0.0]), 'prior': 0.3, 'label': 'towards_B'},
    ]

# -------------------- Run Perception --------------------
final_nodes = perception_tree_of_thought(
    observation=obs_vec,
    initial_hypotheses=initial_hypotheses,
    expand_fn=custom_expand_fn,
    beam_size=beam_size,
    depth=depth,
    lambda_likelihood=lambda_likelihood,
    beta_attention=beta_attention
)

# -------------------- Display Results --------------------
st.subheader("Final Hypotheses")
for n in final_nodes:
    st.write(f"Path: {n.path_labels()}, Depth: {n.depth}, Score: {n.score:.4f}, Attention: {n.attention:.4f}")
