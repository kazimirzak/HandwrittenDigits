import java.io.IOException;

/**
 * @author Kenny Brink - kebri18@student.sdu.dk
 */

public class ImageRecognition {

    // File Directory for MNist data
    private static final String DIRECTORY = System.getProperty("user.dir") + "/MNistData";

    //HyperParameters
    private static final int EPOCHS = 30, BATCH_SIZE = 10;
    private static final double LEARNING_RATE = 3;

    // Data
    private static double[][] trainingImages, testImages;
    private static int[] trainingLabels, testLabels;

    /**
     * @param args the command line arguments
     */

    public static void main(String[] args) {
        System.out.println(System.getProperty("user.dir") + "/MNistData");
            Data reader;
        try {
            reader = new Data(DIRECTORY);
            trainingImages = reader.getTrainingImages();
            trainingLabels =  reader.getTrainingLabels();
            testImages = reader.getTestImages();
            testLabels = reader.getTestLabels();
        } catch(IOException ignored) {
        }

        int[] layout = {784, 30, 10};
        Network net = new Network(layout);
        net.training(trainingImages, trainingLabels, testImages, testLabels, BATCH_SIZE, LEARNING_RATE, EPOCHS);
    }

}
