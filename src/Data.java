import Math.Functions;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * @author Kenny Brink - kebri18@student.sdu.dk
 */

class Data {

    // Data
    private double[][] trainingImages, testImages;
    private int[] trainingLabels, testLabels;

    /**
     * Import data from MNistDataReader and pre-processes it. Turns the bytes into doubles and apply sigmoid to it.
     * directory = where the MNist files are saved.
     * validationSize = number of picture for validation batch.
     * @throws IOException if the files for the data are not found.
     */

    Data(String directory) throws IOException {
        Path path = Paths.get(directory);
        MNistDataReader reader = new MNistDataReader(path);
        getAndProcessData(reader);
    }

    /**
     * Retrieves the data from MNistDataReader and stats the pre-processing.
     * @param reader the reader that is used to get the data.
     * @throws IOException if the files for the data are not found.
     */

    private void getAndProcessData(MNistDataReader reader) throws IOException {
        byte[][] trainingImages = reader.getTrainingImages();
        byte[][] testImages = reader.getTestImages();
        byte[] trainingLabels = reader.getTrainingLabels();
        byte[] testLabels = reader.getTestLabels();
        this.trainingImages = preprocessImages(trainingImages);
        this.testImages = preprocessImages(testImages);
        this.trainingLabels = preprocessLabels(trainingLabels);
        this.testLabels = preprocessLabels(testLabels);
    }

    /**
     * @param images the images to preprocess.
     * @return the processed images which is the original pictures converted to doubles and then the sigmoid function
     * is applied to this.
     */

    private static double[][] preprocessImages(byte[][] images) {
        double[][] result = convertToDouble(images);
        result = applyFunction(result);
        return result;
    }

    /**
     * Converts the given byte array to doubles, however it transforms the bytes into unsigned bytes and then into doubles.
     * @param images the images to convert into doubles.
     * @return the images as doubles.
     */

    private static double[][] convertToDouble(byte[][] images) {
        double[][] result = new double[images.length][images[0].length];
        for(int i = 0; i < result.length; i++) {
            for(int j = 0; j < result[i].length; j++) {
                if(images[i][j] < 0)
                    result[i][j] = 256 + images[i][j];
                else
                    result[i][j] = images[i][j];
            }
        }
        return result;
    }

    /**
     * Applies the sigmoid function to the given array.
     * @param images the images to apply the function to.
     * @return the images with sigmoid applied to them.
     */

    private static double[][] applyFunction(double[][] images) {
        double[][] result = new double[images.length][images[0].length];
        for(int i = 0; i < result.length; i++) {
            for(int j = 0; j < result[i].length; j++) {
                if(images[i][j] == 0)
                    result[i][j] = 0;
                else
                    result[i][j] = Functions.sigmoid(images[i][j]);
            }
        }
        return result;
    }

    /**
     * Converts the given array to an int array
     */

    private static int[] preprocessLabels(byte[] labels) {
        int[] result = new int[labels.length];
        for(int i = 0; i < result.length; i++) {
            result[i] = labels[i];
        }
        return result;
    }

    /**
     * Returns a copy of this trainingImages.
     */

    double[][] getTrainingImages() {
        double[][] result = new double[trainingImages.length][trainingImages[0].length];
        for(int i = 0; i < result.length; i++) {
            System.arraycopy(trainingImages[i], 0, result[i], 0, result[i].length);
        }
        return result;
    }

    /**
     * Returns a copy of this testImages.
     */

    double[][] getTestImages() {
        double[][] result = new double[testImages.length][testImages[0].length];
        for(int i = 0; i < result.length; i++) {
            System.arraycopy(testImages[i], 0, result[i], 0, result[i].length);
        }
        return result;
    }

    /**
     * Returns a copy of this trainingLabels.
     */

    int[] getTrainingLabels() {
        int[] result = new int[trainingLabels.length];
        System.arraycopy(trainingLabels, 0, result, 0, result.length);
        return result;
    }

    /**
     * Returns a copy of this testLabels.
     */

    int[] getTestLabels() {
        int[] result = new int[testLabels.length];
        System.arraycopy(testLabels, 0, result, 0, result.length);
        return result;
    }
}


