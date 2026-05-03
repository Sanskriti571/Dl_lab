import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Input sentence
sentence = ["I", "love", "deep", "learning"]

# 2. Raw attention scores (can be any values)
attention_scores = torch.tensor([0.1, 0.3, 0.4, 0.2])

# 3. Apply softmax to normalize
attention_weights = F.softmax(attention_scores, dim=0).detach().numpy()

# 4. Plot heatmap
plt.figure(figsize=(8, 2))
sns.heatmap(
    [attention_weights],
    annot=True,
    xticklabels=sentence,
    yticklabels=["Attention"],
    cmap="Blues"
)

plt.title("Attention Heatmap Visualization")
plt.xlabel("Words")
plt.show()
