/**
 * @author Kenny Brink - kebri18@student.sdu.dk
 */

public class Matrix {

    /**
     * Matrix matrix multiplication. Returns matrix1 * matrix2.
     * @param <T>
     * @param matrix1
     * @param matrix2
     * @return
     */

    public static double[][] multiplication(double[][] matrix1, double[][] matrix2) {
        int rows1 = matrix1.length;
        int cols1 = matrix1[0].length;
        int rows2 = matrix2.length;
        int cols2 = matrix2[0].length;
        if(cols1 != rows2)
            throw new IllegalArgumentException("Illegal matrix dimensions");
        double[][] result = new double[rows1][cols2];
        for(int i = 0; i < rows1; i++) {
            for(int j = 0; j < cols2; j++) {
                result[i][j] = 0.0;
                for(int k = 0; k < cols1; k++) {
                    result[i][j] = result[i][j] + matrix1[i][k] * matrix2[k][j];
                }
            }
        }
        return result;
    }

    /**
     * Matrix vector multiplication. Returns matrix * vector.
     * @param <T>
     * @param matrix
     * @param vector
     * @return
     */

    public static double[] multiplication(double[][] matrix, double[] vector) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        if(vector.length != cols)
            throw new IllegalArgumentException("Illegal matrix dimensions.");
        double[] result = new double[rows];
        for (int i = 0; i < rows; i++) {
            result[i] = 0.0;
            for (int j = 0; j < cols; j++) {
                result[i] = result[i] + matrix[i][j] * vector[j];
            }
        }
        return result;
    }

    /**
     * Matrix vector multiplication. Returns vector * matrix.
     * @param <T>
     * @param vector
     * @param matrix
     * @return
     */

    public static double[][] multiplication(double[] vector, double[][] matrix) {
        int cols = vector.length;
        int rows = matrix.length;
        if(matrix[0].length != 1)
            throw new IllegalArgumentException("Illegal matrix dimensions.");
        double[][] result = new double[cols][rows];
        for(int i = 0; i < cols; i++) {
            for(int j = 0; j < rows; j++) {
                result[i][j] = matrix[j][0] * vector[i];
            }
        }
        return result;
    }

    /**
     * Dot product of 2 vectors. Returns vector1.vector2.
     * @param <T>
     * @param vector1
     * @param vector2
     * @return
     */

    public static double multiplication(double[] vector1, double[] vector2) {
        if(vector1.length != vector2.length)
            throw new IllegalArgumentException("Illegal vector dimensions.");
        double result = 0.0;
        for (int i = 0; i < vector1.length; i++) {
            result = result + vector1[i] * vector2[i];
        }
        return result;
    }

    /**
     * Constant Multiplication on matrix. Returns matrix * constant.
     * @param <T>
     * @param matrix
     * @param constant
     * @return
     */

    public static double[][] constantMultiplication(double[][] matrix, double constant) {
        double[][] result = new double[matrix.length][matrix[0].length];
        for(int i = 0; i < result.length; i++) {
            for(int j = 0; j < result[i].length; j++) {
                result[i][j] = matrix[i][j] * constant;
            }
        }
        return result;
    }

    /**
     * Constant Multiplication on vector. Returns vector * constant.
     * @param <T>
     * @param vector
     * @param constant
     * @return
     */

    public static double[] constantMultiplication(double[] vector, double constant) {
        double[] result = new double[vector.length];
        for(int i = 0; i < result.length; i++) {
            result[i] = vector[i] * constant;
        }
        return result;
    }

    /**
     * Tranposes a matrix. Returns matrix^T.
     * @param <T>
     * @param matrix
     * @return
     */

    public static double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }

    /**
     * Transposes a vector. Returns vector^T.
     * @param <T>
     * @param vector
     * @return
     */

    public static double[][] transpose(double[] vector) {
        int rows = 1;
        int cols = vector.length;
        double[][] result = new double[cols][rows];
        for(int i = 0; i < cols; i++) {
            result[i][0] = vector[i];
        }
        return result;
    }

    /**
     * Matrix matrix addition. Returns matrix1 + matrix2.
     * @param <T>
     * @param matrix1
     * @param matrix2
     * @return
     */

    public static double[][] add(double[][] matrix1, double[][] matrix2){
        if(matrix1.length != matrix2.length)
            throw new IllegalArgumentException("Illegal vector dimensions. " + matrix1.length + " != " + matrix2.length);
        int rows = matrix1.length;
        int cols = matrix1[0].length;
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = matrix1[i][j] + matrix2[i][j];
            }
        }
        return result;
    }

    /**
     * Vector vector addition. Returns vector1 + vector2.
     * @param <T>
     * @param vector1
     * @param vector2
     * @return
     */

    public static double[] add(double[] vector1, double[] vector2) {
        int rows = vector1.length;
        double[] result = new double[rows];
        for(int i = 0; i < rows; i++) {
            result[i] = vector1[i] + vector2[i];
        }
        return result;
    }

    /**
     * Matrix matrix subtraction. Returns matrix1 - matrix2.
     * @param <T>
     * @param matrix1
     * @param matrix2
     * @return
     */

    public static double[][] subtract(double[][] matrix1, double[][] matrix2) {
        int rows = matrix1.length;
        int cols = matrix1[0].length;
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = matrix1[i][j] - matrix2[i][j];
            }
        }
        return result;
    }

    /**
     * Vector vector subtraction. Returns vector1 - vector2.
     * @param <T>
     * @param vector1
     * @param vector2
     * @return
     */

    public static double[] subtract(double[] vector1, double[] vector2) {
        int rows = vector1.length;
        int cols = 1;
        double[] result = new double[rows];
        for(int i = 0; i < rows; i++) {
            result[i] = vector1[i] - vector2[i];
        }
        return result;
    }


    public static double[] hadamardProduct(double[] vector1, double[] vector2) {
        int rows = vector1.length;
        if(vector1.length != vector2.length)
            throw new IllegalArgumentException("Illegal vector dimensions. " + vector1.length + " != " + vector2.length);
        double[] result = new double[rows];
        for(int i = 0; i < rows; i++) {
            result[i] = vector1[i] * vector2[i];
        }
        return result;
    }

    public static double[][] hadamardProduct(double[][] matrix1, double[][] matrix2) {
        int rows1 = matrix1.length;
        int cols1 = matrix1[0].length;
        int rows2 = matrix2.length;
        int cols2 = matrix2[0].length;
        if(rows1 != rows2 && cols1 != cols2)
            throw new IllegalArgumentException("Illegal matrix dimensions.");
        double[][] result = new double[rows1][cols1];
        for(int i = 0; i < rows1; i++) {
            for(int j = 0; j < cols1; j++) {
                result[i][j] = matrix1[i][j] * matrix2[i][j];
            }
        }
        return result;
    }

    /**
     * Textual representation of a matrix.
     * @param <T>
     * @param matrix
     * @return
     */

    public static String matrixToString(double[][] matrix) {
        String result = "[";
        for(int i = 0; i < matrix.length; i++) {
            result += "[";
            for(int j = 0; j < matrix[i].length; j++) {
                result += matrix[i][j];
                if(j != matrix[i].length - 1)
                    result += ", ";
            }
            if(i != matrix.length - 1)
                result+= "]\n ";
        }
        result+= "]] ";
        return result;
    }

    /**
     * Textual representation of a vector.
     * @param <T>
     * @param vector
     * @return
     */

    public static String matrixToString(double[] vector) {
        String result = "[";
        for(int i = 0; i < vector.length; i++) {
            result += vector[i];
            if(i != vector.length - 1)
                result += ", ";
        }
        result += "]";
        return result;
    }
}
