# Candidate-Elimination Algorithm

# Training data directly included in the program
data = [
    ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", "Yes"],
    ["Sunny", "Warm", "High", "Strong", "Warm", "Same", "Yes"],
    ["Rainy", "Cold", "High", "Strong", "Warm", "Change", "No"],
    ["Sunny", "Warm", "High", "Strong", "Cool", "Change", "Yes"]
]

# Attribute names
attributes = [
    "Sky",
    "AirTemp",
    "Humidity",
    "Wind",
    "Water",
    "Forecast"
]

num_attributes = len(attributes)

# Most Specific hypothesis
S = ["0"] * num_attributes

# Most General hypothesis
G = [["?"] * num_attributes]


# Check whether a hypothesis covers an example
def covers(hypothesis, example):
    for h, x in zip(hypothesis, example):
        if h != "?" and h != x:
            return False
    return True


# Check whether h1 is more general than h2
def more_general(h1, h2):
    return all(
        a == "?" or a == b or b == "0"
        for a, b in zip(h1, h2)
    )


# Generalize Specific boundary
def generalize_S(S, example):

    new_S = S.copy()

    for i in range(num_attributes):

        if new_S[i] == "0":
            new_S[i] = example[i]

        elif new_S[i] != example[i]:
            new_S[i] = "?"

    return new_S


# Find possible values for each attribute
domains = []

for i in range(num_attributes):

    values = set()

    for row in data:
        values.add(row[i])

    domains.append(values)


# Specialize General boundary
def specialize_G(G, example):

    new_G = []

    for hypothesis in G:

        if covers(hypothesis, example):

            for i in range(num_attributes):

                if hypothesis[i] == "?":

                    for value in domains[i]:

                        if value != example[i]:

                            new_hypothesis = hypothesis.copy()
                            new_hypothesis[i] = value

                            new_G.append(new_hypothesis)

        else:
            new_G.append(hypothesis)

    return new_G


# -------------------------------
# Candidate-Elimination Algorithm
# -------------------------------

for row in data:

    example = row[:-1]
    target = row[-1]

    print("\nExample:", example)
    print("Target:", target)

    # Positive example
    if target == "Yes":

        # Remove G hypotheses that do not cover example
        G = [
            g for g in G
            if covers(g, example)
        ]

        # Generalize S
        if not covers(S, example):
            S = generalize_S(S, example)

    # Negative example
    else:

        # Specialize G
        G = specialize_G(G, example)

        # Keep only hypotheses more general than S
        G = [
            g for g in G
            if more_general(g, S)
        ]

    print("S =", S)
    print("G =", G)


# -------------------------------
# Final Result
# -------------------------------

print("\n==============================")
print("FINAL RESULT")
print("==============================")

print("\nSpecific Boundary (S):")
print(S)

print("\nGeneral Boundary (G):")

for g in G:
    print(g)