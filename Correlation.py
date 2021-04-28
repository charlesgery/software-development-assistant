class Correlation:

    @staticmethod
    def basic_correlation(number_modifications_same_commit, number_modificationsA):

        return 100 * number_modifications_same_commit / number_modificationsA

    @staticmethod
    def addition_correlation(number_modifications_same_commit, number_modificationsA, number_modificationsB, alpha):

        node1_correlation = number_modifications_same_commit / number_modificationsA
        node2_correlation = number_modifications_same_commit / number_modificationsB
        correlation = 100 * ((1 + alpha) * node1_correlation - alpha * node2_correlation) / (1 + alpha)

        return correlation

    @staticmethod
    def multiplication_correlation(number_modifications_same_commit, number_modificationsA, number_modificationsB, alpha):

        node1_correlation = number_modifications_same_commit / number_modificationsA
        node2_correlation = number_modifications_same_commit / number_modificationsB
        correlation = 100 * node1_correlation * min(1, 1 + alpha * (node1_correlation - node2_correlation))

        return correlation
    