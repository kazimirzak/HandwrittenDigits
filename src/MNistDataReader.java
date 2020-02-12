import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Path;

/**
 * @author Kenny Brink - kebri18@student.sdu.dk
 */

class MNistDataReader {

    // File names.
    private static final String TRAIN_IMAGE = "train-images.idx3-ubyte",
                                TRAIN_LABEL = "train-labels.idx1-ubyte",
                                TEST_IMAGE = "t10k-images.idx3-ubyte",
                                TEST_LABEL = "t10k-labels.idx1-ubyte";

    // Paths for images and labels.
    private static Path TRAIN_IMAGE_DIRECTORY,
                        TRAIN_LABEL_DIRECTORY,
                        TEST_IMAGE_DIRECTORY,
                        TEST_LABEL_DIRECTORY;

    // Streams to get data from.
    private DataInputStream trainImageStream,
                            trainLabelStream,
                            testImageStream,
                            testLabelStream;

    // Constants aquired through the data streams.

    private int numOfTrainImages, numOfTrainLabels,      numOfTestImages,
                numOfTestLabels,  numOfRows,             numOfCols;

    /**
     * Constructs the data reader and initializes it so its ready to give images.
     * @param dataDirectory The directory of the MNist Data files
     * @throws IOException if file is not found or if wrong file is found.
     */

    MNistDataReader(Path dataDirectory) throws IOException {
        TRAIN_IMAGE_DIRECTORY = dataDirectory.resolve(TRAIN_IMAGE);
        TRAIN_LABEL_DIRECTORY = dataDirectory.resolve(TRAIN_LABEL);
        TEST_IMAGE_DIRECTORY = dataDirectory.resolve(TEST_IMAGE);
        TEST_LABEL_DIRECTORY = dataDirectory.resolve(TEST_LABEL);
        init();
    }

    /**
     * Initializes the streams.
     * @throws IOException if file is not found or if wrong file is found.
     */

    private void init() throws IOException {
        trainImageStream = new DataInputStream(new FileInputStream(TRAIN_IMAGE_DIRECTORY.toFile()));
        trainLabelStream = new DataInputStream(new FileInputStream(TRAIN_LABEL_DIRECTORY.toFile()));
        testImageStream = new DataInputStream(new FileInputStream(TEST_IMAGE_DIRECTORY.toFile()));
        testLabelStream = new DataInputStream(new FileInputStream(TEST_LABEL_DIRECTORY.toFile()));
        magicHeaders();
        getConstants();
    }

    /**
     * Removes the magic headers from all 4 files and checks if they are correct.
     * Magic header for training images: 2051 or 0x803.
     * Magic header for training labels: 2049 or 0x801.
     * Magic header for test images: 20551 or 0x803.
     * Magic header for test labels: 2049 or 0x801.
     * @throws IOException if magic header is wrong.
     */

    private void magicHeaders() throws IOException {
        int trainImageHeader = trainImageStream.readInt();
        int trainLabelHeader = trainLabelStream.readInt();
        int testImageHeader = testImageStream.readInt();
        int testLabelHeader = testLabelStream.readInt();
        if(trainImageHeader != 0x803)
            throw new IOException("Expected magic header \"0x803\" for training images but received " + trainImageHeader);
        if(trainLabelHeader != 0x801)
            throw new IOException("Expected magic header \"0x801\" for training labels but received " + trainLabelHeader);
        if(testImageHeader != 0x803)
            throw new IOException("Expected magic header \"0x803\" for test images but received " + testImageHeader);
        if(testLabelHeader != 0x801)
            throw new IOException("Expected magic header \"0x801\" for test labels but received " + testLabelHeader);
    }

    /**
     * Removes and saves the constants saved in the files.
     * @throws IOException standard exceptions.
     */

    private void getConstants() throws IOException {
        numOfTrainImages = trainImageStream.readInt();
        numOfTrainLabels = trainLabelStream.readInt();
        numOfTestImages = testImageStream.readInt();
        numOfTestLabels = testLabelStream.readInt();
        numOfCols = trainImageStream.readInt();
        numOfRows = trainImageStream.readInt();
        //This is just to remove the number of rows and columns from the testImageStream as they are not needed.
        testImageStream.readInt();
        testImageStream.readInt();
    }

    /**
     * Returns a byte array of all the training imgaes.
     * @throws IOException standard exceptions.
     */

    byte[][] getTrainingImages() throws IOException {
        byte[][] result = new byte[numOfTrainImages][numOfRows * numOfCols];
        for(int i = 0; i < numOfTrainImages; i++) {
            trainImageStream.read(result[i]);
        }
        return result;
    }

    /**
     * Returns a byte array with training labels that match the images.
     * @throws IOException standard exceptions.
     */

    byte[] getTrainingLabels() throws IOException {
        byte[] result = new byte[numOfTrainLabels];
        trainLabelStream.read(result);
        return result;
    }

    /**
     * Returns a byte array with the test images.
     * @throws IOException standard exceptions.
     */

    byte[][] getTestImages() throws IOException {
        byte[][] result = new byte[numOfTestImages][numOfRows * numOfCols];
        for(int i = 0; i < numOfTestImages; i++) {
            testImageStream.read(result[i]);
        }
        return result;
    }

    /**
     * Returns a byte array with the labels for the test images.
     * @throws IOException standard exceptions.
     */

    byte[] getTestLabels() throws IOException {
        byte[] result = new byte[numOfTestLabels];
        testLabelStream.read(result);
        return result;
    }
}
