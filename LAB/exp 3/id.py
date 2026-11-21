import math
import pandas as pd

# -----------------------------
# Play Tennis Dataset
# -----------------------------
data = {
    "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain",
                "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast",
                "Overcast", "Rain"],
    
    "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool",
                    "Cool", "Mild", "Cool", "Mild", "Mild", "Mild",
                    "Hot", "Mild"],
    
    "Humidity": ["High", "High", "High", "High", "Normal", "Normal",
                 "Normal", "High", "Normal", "Normal", "Normal", "High",
                 "Normal", "High"],
    
    "Wind": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong",
             "Strong", "Weak", "Weak", "Weak", "Strong", "Strong",
             "Weak", "Strong"],
    
    "PlayTennis": ["No", "No", "Yes", "Yes", "Yes", "No",
                   "Yes", "No", "Yes", "Yes", "Yes", "Yes",
                   "Yes", "No"]
}

df = pd.DataFrame(data)

# -----------------------------
# Calculate Entropy
# -----------------------------
def entropy(data):
    values = data.value_counts()
    total = len(data)

    ent = 0

    for count in values:
        probability = count / total
        ent -= probability * math.log2(probability)

    return ent


# -----------------------------
# Calculate Information Gain
# -----------------------------
def information_gain(data, attribute, target):
    total_entropy = entropy(data[target])

    weighted_entropy = 0

    for value in data[attribute].unique():

        subset = data[data[attribute] == value]

        probability = len(subset) / len(data)

        weighted_entropy += probability * entropy(subset[target])

    return total_entropy - weighted_entropy


# -----------------------------
# ID3 Algorithm
# -----------------------------
def id3(data, attributes, target):

    # If all target values are same
    if len(data[target].unique()) == 1:
        return data[target].iloc[0]

    # If no attributes remain
    if len(attributes) == 0:
        return data[target].mode()[0]

    # Find attribute with highest Information Gain
    gains = {}

    for attribute in attributes:
        gains[attribute] = information_gain(data, attribute, target)

    best_attribute = max(gains, key=gains.get)

    tree = {best_attribute: {}}

    remaining_attributes = [
        attribute for attribute in attributes
        if attribute != best_attribute
    ]

    # Create branches
    for value in data[best_attribute].unique():

        subset = data[data[best_attribute] == value]

        if len(subset) == 0:
            tree[best_attribute][value] = data[target].mode()[0]

        else:
            tree[best_attribute][value] = id3(
                subset,
                remaining_attributes,
                target
            )

    return tree


# -----------------------------
# Build Decision Tree
# -----------------------------
attributes = ["Outlook", "Temperature", "Humidity", "Wind"]

tree = id3(df, attributes, "PlayTennis")

print("Decision Tree:")
print(tree)


# -----------------------------
# Classify New Sample
# -----------------------------
def classify(tree, sample):

    if not isinstance(tree, dict):
        return tree

    attribute = next(iter(tree))

    value = sample[attribute]

    branch = tree[attribute][value]

    return classify(branch, sample)


# New sample
new_sample = {
    "Outlook": "Sunny",
    "Temperature": "Cool",
    "Humidity": "High",
    "Wind": "Strong"
}

result = classify(tree, new_sample)

print("\nNew Sample:")
print(new_sample)

print("\nClassification:")
print("Play Tennis =", result)