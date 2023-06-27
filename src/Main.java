import ws.schild.jave.Encoder;
import ws.schild.jave.EncoderException;
import ws.schild.jave.MultimediaObject;
import ws.schild.jave.encode.AudioAttributes;
import ws.schild.jave.encode.EncodingAttributes;
import java.io.*;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class Main {
    private static final int AUDIO_SNIPPET_LENGTH = 5;
    private File csvFile;
    public Main() {
        File outputDir = new File("output/");
        if(!outputDir.exists()) {
            boolean createdFolderSuccessfully = outputDir.mkdir();
            if(!createdFolderSuccessfully) {
                throw new RuntimeException("Could not create output folder!");
            }
        }
        File inputDir = new File("input/");
        if(!inputDir.exists()) {
            throw new IllegalArgumentException("No input available!");
        }
        try {
            createCSV();
        } catch(IOException e) {
            e.printStackTrace();
        }
    }

    private void createCSV() throws IOException {
        csvFile = new File("data.csv");
        try (FileWriter fileWriter = new FileWriter(csvFile)){
            fileWriter.write("filename,mfcc1_mean,mfcc1_std,mfcc2_mean,mfcc2_std,mfcc3_mean,mfcc3_std,mfcc4_mean,mfcc4_std,mfcc5_mean,mfcc5_std,mfcc6_mean,mfcc6_std,mfcc7_mean,mfcc7_std,mfcc8_mean,mfcc8_std,mfcc9_mean,mfcc9_std,mfcc10_mean,mfcc10_std,mfcc11_mean,mfcc11_std,mfcc12_mean,mfcc12_std,mfcc13_mean,mfcc13_std,mfcc14_mean,mfcc14_std,mfcc15_mean,mfcc15_std,mfcc16_mean,mfcc16_std,mfcc17_mean,mfcc17_std,mfcc18_mean,mfcc18_std,mfcc19_mean,mfcc19_std,mfcc20_mean,mfcc20_std,label\n");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        processInputDirectory();
        processOutputDirectory();
    }
    //This method goes through every directory in the output thread and calls a method
    //which generates the features for the .csv and writes them
    private void processOutputDirectory() {
        File wavDirectory = new File("output/");
        for(File genreDirectory : Objects.requireNonNull(wavDirectory.listFiles())) {
            if(genreDirectory.isDirectory()) {
                processSongs(genreDirectory);
            }

        }
    }
    private void processSongs(File f) {
        for(File songfile : Objects.requireNonNull(f.listFiles())) {
            ArrayList<String> featuresList = generateMFCCsFromFile(songfile, f.getName());
            if(featuresList != null) {
                writeFeaturesToCsv(featuresList);
            }
        }
    }
    private void writeFeaturesToCsv(ArrayList<String> featuresList) {
        for(String s : featuresList) {
            try (BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(csvFile, true))){
                bufferedWriter.append(s);
                bufferedWriter.flush();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }
    public static void main(String[] args) {
        new Main();
    }
    private ArrayList<String> generateMFCCsFromFile(File f, String genre) {
        ExecutorService executor = Executors.newFixedThreadPool(20);
        ArrayList<String> csvStringArrayList = new ArrayList<>();
        JLibrosa librosa = new JLibrosa();
        float[] magValues;
        long songLength;
        try {
            songLength = WavFile.openWavFile(f).getDuration();
            magValues = librosa.loadAndRead(f.getPath(), -1,-1);
        } catch (IOException | WavFileException e) {
            e.printStackTrace();
            return null;
        } catch (FileFormatNotSupportedException e) {
            throw new RuntimeException(e);
        }
        for(int i = 0; i+AUDIO_SNIPPET_LENGTH <= songLength; i=i+AUDIO_SNIPPET_LENGTH) {
            int finalI = i;
            executor.execute(() -> {
                String filename = f.getName().substring(0,f.getName().length()-4)+ finalI +"s_to"+(finalI+AUDIO_SNIPPET_LENGTH)+"s.wav,";
                float[][] mfccValues = librosa.generateMFCCFeatures(Arrays.copyOfRange(magValues, finalI * librosa.getSampleRate(), (finalI + AUDIO_SNIPPET_LENGTH) * librosa.getSampleRate()),librosa.getSampleRate(),20);
                float[] meanMfcc = librosa.generateMeanMFCCFeatures(mfccValues, mfccValues.length, mfccValues[0].length);
                double[] standardDeviations = generateStandardDeviations(mfccValues);
                double[] csvArray = createArrayForCSV(meanMfcc, standardDeviations);
                String csvArrayAsString = Arrays.toString(csvArray);
                csvArrayAsString = csvArrayAsString.substring(1, csvArrayAsString.length()-2);
                csvArrayAsString = filename+csvArrayAsString+","+genre+"\n";
                csvStringArrayList.add(csvArrayAsString);
            });
        }
        executor.shutdown();
        boolean terminatedSuccessfully;
        try {
            terminatedSuccessfully = executor.awaitTermination(300, TimeUnit.SECONDS);

        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        if(!terminatedSuccessfully) {
            throw new RuntimeException("An unexpected error has occurred");
        }
        return csvStringArrayList;
    }
    //This method takes the mfcc values and generates the standard deviations of every 2nd dimension
    private double[] generateStandardDeviations(float[][] mfccValues) {
        double[] standardDeviations = new double[mfccValues.length];
        for(int i = 0; i < mfccValues.length; i++) {
            standardDeviations[i] = getStandardDeviation(mfccValues[i]);
        }
        return standardDeviations;
    }
    //This method merges two arrays into one, as per the requirement for the .csv to have mfcc_mean[0], mfcc_std[0], etc.
    private double[] createArrayForCSV(float[] meanMFCC, double[] deviations) {
        double[] csvArray = new double[meanMFCC.length*2];
        int apiArrayPointer = 0;
        for(int i = 0; i < meanMFCC.length; i++) {
            csvArray[apiArrayPointer] = meanMFCC[i];
            apiArrayPointer++;
            csvArray[apiArrayPointer] = deviations[i];
            apiArrayPointer++;
        }
        return csvArray;
    }
    private double getStandardDeviation(float[] array) {
        // get the sum of array
        double sum = 0.0;
        for (double i : array) {
            sum += i;
        }
        int length = array.length;
        double mean = sum / length;
        // calculate the standard deviation
        double standardDeviation = 0.0;
        for (double num : array) {
            standardDeviation += Math.pow(num - mean, 2);
        }
        return Math.sqrt(standardDeviation / length);
    }
    //This method looks at the input directory, takes all subdirectories and recreates them in the output directory
    //It also takes every file from the input directory and either converts it, it is .mp3
    //Or it copies it, if it's a .wav
    private void processInputDirectory() throws IOException {
        File inputDirectory = new File("input/");
        if(inputDirectory.listFiles() == null) {
            throw new IOException("Input Directory could not be read!");
        }
        //this loop iterates over all subdirectories of the input directory
        for(File directory : Objects.requireNonNull(inputDirectory.listFiles(File::isDirectory))) {
            File outputDirectory = new File("output/"+directory.getName());
            System.out.println("Created the "+outputDirectory.getName()+" directory in output dir.");
            if(!outputDirectory.exists()) {
                if(!outputDirectory.mkdir()) {
                    throw new RuntimeException("Could not create directory.");
                }
            }
            convertFilesFromDirectory(directory, outputDirectory);
        }
    }
    //This method iterates over a directory and converts a file if it's .mp3 or just copies it if it's a .wav
    //Everything else gets ignored
    private void convertFilesFromDirectory(File directory, File outputDirectory) throws IOException {
        for(File f : Objects.requireNonNull(directory.listFiles())) { //iterate over all files
            if(f.getName().endsWith(".mp3")) { //if file ends on .mp3
                try {
                    convertMp3ToWav(f, outputDirectory.getPath()+"/");
                } catch (EncoderException e) {
                    e.printStackTrace();
                }
            } else {
                if(!f.isDirectory() && f.getName().endsWith(".wav")) { //if the file is not yet another subdirectory, and is .wav, it should get copied
                    File copy = new File(outputDirectory.getPath()+"/"+f.getName()); //where the copy should be, this is necessary for the .toPath() function
                    if(!copy.exists()) {
                        Files.copy(f.toPath(), copy.toPath()); //copy the file if it is already a .wav
                    }
                }
            }
        }
    }
    //This method converts the mp3 file (input) to a .wav file (with the output path)
    public static void convertMp3ToWav(File input, String outputPath) throws EncoderException {
        Encoder encoder = new Encoder();
        MultimediaObject object = new MultimediaObject(input);
        File output = new File(outputPath+input.getName().substring(0,input.getName().length()-4)+".wav");
        AudioAttributes audioAttributes = new AudioAttributes();
        audioAttributes.setCodec("pcm_s16le");
        audioAttributes.setBitRate(object.getInfo().getAudio().getBitRate());
        audioAttributes.setChannels(1);
        audioAttributes.setSamplingRate(object.getInfo().getAudio().getSamplingRate());
        EncodingAttributes encodingAttributes = new EncodingAttributes();
        encodingAttributes.setOutputFormat("wav");
        encodingAttributes.setAudioAttributes(audioAttributes);
        encoder.encode(object,output,encodingAttributes);
        System.out.println("Created: "+output.getName());
    }
}