def calculate_ev(probability, odds):

    implied_probability = 1 / odds

    ev = (
        probability * odds
    ) - 1

    return {
        "probability": probability,
        "implied_probability": implied_probability,
        "ev": ev
    }

# ejemplo

result = calculate_ev(
    probability=0.65,
    odds=1.90
)

print(result)
