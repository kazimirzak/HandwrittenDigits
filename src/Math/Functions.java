package Math;

/**
 * @author Kenny Brink - kebri18@student.sdu.dk
 */

public class Functions {

    /**
     * The sigmoid function.
     */

    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(((-1.0) * x)));
    }

    /**
     * Derivative of the sigmoid function.
     */

    public static double sigmoidPrime(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }
}
