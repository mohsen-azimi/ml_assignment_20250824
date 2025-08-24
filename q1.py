
def prob_rain_at_least_n(ps, n):
    N = len(ps)

    # skipping edge cases, and validating inputs

    dp = [0.0] * (N + 1)  # dp[k] = chance of k rainy days from first i days
    dp[0] = 1.0  # base: we have 0 rainy days with certainty at the start

    for p in ps:
        for k in range(N, -1, -1):  # from right-to-left; so we don't overwrite dp[k-1]
            # today is dry, count stays k + today is rainy, count goes from k-1 to k
            dp[k] = (dp[k] * (1.0 - p)) + ((dp[k - 1] * p) if k > 0 else 0.0)
    return sum(dp[n:])


def prob_rain_more_than_n(p, n):
    return prob_rain_at_least_n(p, n + 1)


# Example: 40% chance every day, 150 days of rain
print("P(K >= 150) =", prob_rain_at_least_n([0.40] * 365, 150))
print("P(K > 150)  =", prob_rain_more_than_n([0.40] * 365, 150))
