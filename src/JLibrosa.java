import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import org.apache.commons.math3.complex.Complex;


/**
 *
 * This Class is an equivalent of Python Librosa utility used to extract the Audio features from given Wav file.
 *
 * @author abhi-rawat1
 *
 */
public class JLibrosa {
	private final int BUFFER_SIZE = 4096;
	private int noOfFrames = -1;
	private int sampleRate = -1;
	private int noOfChannels = -1;

	private double fMax = 44100 / 2.0;
	private double fMin = 0.0;
	private final int n_fft = 2048;
	private final int hop_length = 512;
	private final int n_mels = 128;
	private long duration = 7;
	
	
	public double getfMax() {
		return fMax;
	}



	public double getfMin() {
		return fMin;
	}



	public int getN_fft() {
		return n_fft;
	}



	public int getHop_length() {
		return hop_length;
	}



	public int getN_mels() {
		return n_mels;
	}



	public int getNoOfChannels() {
		return noOfChannels;
	}



	public void setNoOfChannels(int noOfChannels) {
		this.noOfChannels = noOfChannels;
	}



	public JLibrosa() {

	}

	

	public int getNoOfFrames() {
		return noOfFrames;
	}



	public void setNoOfFrames(int noOfFrames) {
		this.noOfFrames = noOfFrames;
	}



	public int getSampleRate() {
		return sampleRate;
	}



	public void setSampleRate(int sampleRate) {
		this.sampleRate = sampleRate;
		this.fMax = sampleRate/2.0;
	}

	public float[][] loadAndReadAcrossChannelsWithOffset(String path, int sr, int readDurationInSec, int offsetDuration)
			throws IOException, WavFileException, FileFormatNotSupportedException {
		return readMagnitudeValuesFromFile(path, sr, readDurationInSec, offsetDuration);
	}
	


	public float[][] loadAndReadAcrossChannels(String path, int sr, int readDurationInSec)
			throws IOException, WavFileException, FileFormatNotSupportedException {
		return loadAndReadAcrossChannelsWithOffset(path, sr, readDurationInSec, 0);
	}

	private float[][] readMagnitudeValuesFromFile(String path, int sampleRate, int readDurationInSeconds, int offsetDuration)
			throws IOException, WavFileException, FileFormatNotSupportedException {

		if(!path.endsWith(".wav")) {
			throw new FileFormatNotSupportedException("File format not supported. jLibrosa currently supports audio processing of only .wav files");
		}

		File sourceFile = new File(path);
		WavFile wavFile = null;

		wavFile = WavFile.openWavFile(sourceFile);
		int mNumFrames = (int) (wavFile.getNumFrames());
		int mSampleRate = (int) wavFile.getSampleRate();
		int mChannels = wavFile.getNumChannels();

		int totalNoOfFrames = mNumFrames;
		int frameOffset = offsetDuration * mSampleRate;
		int tobeReadFrames = readDurationInSeconds * mSampleRate;

		if(tobeReadFrames > (totalNoOfFrames - frameOffset)) {
			tobeReadFrames = totalNoOfFrames - frameOffset;
		}

		if (readDurationInSeconds != -1) {
			mNumFrames = tobeReadFrames;
			wavFile.setNumFrames(mNumFrames);
		}


		this.setNoOfChannels(mChannels);
		this.setNoOfFrames(mNumFrames);
		this.setSampleRate(mSampleRate);


		if (sampleRate != -1) {
			mSampleRate = sampleRate;
		}
		// Read the magnitude values across both the channels and save them as part of
		// multi-dimensional array

		float[][] buffer = new float[mChannels][mNumFrames];
		long readFrameCount = 0;
		//for (int i = 0; i < loopCounter; i++) {

		readFrameCount = wavFile.readFrames(buffer, mNumFrames, frameOffset);

		//}
		duration = wavFile.getDuration();

		wavFile.close();


		return buffer;

	}
	public long getDuration() {
		return duration;
	}
	public float[][] generateMFCCFeatures(float[] magValues, int mSampleRate, int nMFCC, int n_fft, int n_mels, int hop_length) {

		AudioFeatureExtraction mfccConvert = new AudioFeatureExtraction();
		
		mfccConvert.setN_mfcc(nMFCC);
		mfccConvert.setN_mels(n_mels);
		mfccConvert.setHop_length(hop_length);
		
		if(mSampleRate==-1) {
			mSampleRate = this.getSampleRate();
		}
		
		mfccConvert.setSampleRate(mSampleRate);
		mfccConvert.setN_mfcc(nMFCC);
		float [] mfccInput = mfccConvert.extractMFCCFeatures(magValues); //extractMFCCFeatures(magValues);
		
		int nFFT = mfccInput.length / nMFCC;
		float[][] mfccValues = new float[nMFCC][nFFT];

		// loop to convert the mfcc values into multi-dimensional array
		for (int i = 0; i < nFFT; i++) {
			int indexCounter = i * nMFCC;
			int rowIndexValue = i % nFFT;
			for (int j = 0; j < nMFCC; j++) {
				mfccValues[j][rowIndexValue] = mfccInput[indexCounter];
				indexCounter++;
			}
		}

		return mfccValues;
		
	}
	
	
	
	
	/**
	 * This function calculates and returns the MFCC values of given Audio Sample
	 * values.
	 * 
	 * @param magValues
	 * @param nMFCC
	 * @return
	 */
	public float[][] generateMFCCFeatures(float[] magValues, int mSampleRate, int nMFCC) {


		return this.generateMFCCFeatures(magValues, mSampleRate, nMFCC, this.n_fft, this.n_mels, this.hop_length);
		
	}

	/**
	 * This function calculates and return the Mean MFCC values.
	 * 
	 * @param mfccValues
	 * @param nMFCC
	 * @param nFFT
	 * @return
	 */
	public float [] generateMeanMFCCFeatures(float[][] mfccValues, int nMFCC, int nFFT) {
		// code to take the mean of mfcc values across the rows such that
		// [nMFCC x nFFT] matrix would be converted into
		// [nMFCC x 1] dimension - which would act as an input to tflite model
		
		
		float [] meanMFCCValues = new float[nMFCC];
		for (int i=0; i<mfccValues.length; i++) {
	        
			float [] floatArrValues = mfccValues[i];
			DoubleStream ds = IntStream.range(0, floatArrValues.length)
                    .mapToDouble(k -> floatArrValues[k]);
			
	        double avg = ds.average().getAsDouble();
	        float floatVal = (float)avg;
	        meanMFCCValues[i] = floatVal;
	    }   

		/*for (int p = 0; p < nMFCC; p++) {
			double fftValAcrossRow = 0;
			for (int q = 0; q < nFFT; q++) {
				fftValAcrossRow = fftValAcrossRow + mfccValues[p][q];
			}
			double fftMeanValAcrossRow = fftValAcrossRow / nFFT;
			meanMFCCValues[p] = (float) fftMeanValAcrossRow;
		} */
		
		return meanMFCCValues;
	}

	/**
	 * This function calculates and returns the melspectrogram of given Audio Sample
	 * values.
	 * 
	 * @param yValues - audio magnitude values
	 * @return
	 */
	public double[][] generateMelSpectroGram(float[] yValues){
		
		AudioFeatureExtraction mfccConvert = new AudioFeatureExtraction();
		return mfccConvert.melSpectrogram(yValues);
	}

	public float[][] generateMelSpectroGram(float[] yValues, int mSampleRate, int n_fft, int n_mels, int hop_length){
		AudioFeatureExtraction mfccConvert = new AudioFeatureExtraction();
		mfccConvert.setSampleRate(mSampleRate);
		mfccConvert.setN_fft(n_fft);
		mfccConvert.setN_mels(n_mels);
		mfccConvert.setHop_length(hop_length);
		return mfccConvert.melSpectrogramWithComplexValueProcessing(yValues);
	}
	
	
	
	/**
	 * This function calculates and returns the STFT values of given Audio Sample
	 * values. STFT stands for Short Term Fourier Transform
	 * 
	 * @param magValues
	 * @param nMFCC
	 * @return
	 */
	public Complex [][] generateSTFTFeatures(float[] magValues, int mSampleRate, int nMFCC, int n_fft, int n_mels, int hop_length) {
		return this.generateSTFTFeaturesWithPadOption(magValues, mSampleRate, nMFCC, n_fft, n_mels, hop_length, true);
	}

	
	
	/**
	 * This function calculates and returns the STFT values of given Audio Sample
	 * values with/without applying padding as one of the argument flag. STFT stands for Short Term Fourier Transform
	 * 
	 * @param magValues
	 * @param nMFCC
	 * @return
	 */
	public Complex [][] generateSTFTFeaturesWithPadOption(float[] magValues, int mSampleRate, int nMFCC, int n_fft, int n_mels, int hop_length, boolean paddingFlag) {
		AudioFeatureExtraction featureExtractor = new AudioFeatureExtraction();
		featureExtractor.setN_fft(n_fft);
		featureExtractor.setN_mels(n_mels);
		featureExtractor.setHop_length(hop_length);
		
		if(mSampleRate == -1) {
			mSampleRate = this.getSampleRate();
		}
		
		featureExtractor.setSampleRate(mSampleRate);
		featureExtractor.setN_mfcc(nMFCC);
		return featureExtractor.extractSTFTFeaturesAsComplexValues(magValues, paddingFlag);
	}
	
	
	
	

	public float [] generateInvSTFTFeatures(Complex [][] stftValues, int mSampleRate, int nMFCC, int n_fft, int n_mels, int hop_length) {
		return this.generateInvSTFTFeaturesWithPadOption(stftValues, mSampleRate, nMFCC, n_fft, n_mels, hop_length, -1, false);
	}
	
	

	public float [] generateInvSTFTFeatures(Complex [][] stftValues, int mSampleRate, int nMFCC, int n_fft, int n_mels, int hop_length, int length) {
		return this.generateInvSTFTFeaturesWithPadOption(stftValues, mSampleRate, nMFCC, n_fft, n_mels, hop_length, length, false);
	}

	public float [] generateInvSTFTFeaturesWithPadOption(Complex [][] stftValues, int mSampleRate, int nMFCC, int n_fft, int n_mels, int hop_length, int length, boolean paddingFlag) {
		AudioFeatureExtraction featureExtractor = new AudioFeatureExtraction();
		featureExtractor.setN_fft(n_fft);
		featureExtractor.setN_mels(n_mels);
		featureExtractor.setHop_length(hop_length);
		featureExtractor.setLength(length);
		
		if(mSampleRate == -1) {
			mSampleRate = this.getSampleRate();
		}
		
		featureExtractor.setSampleRate(mSampleRate);
		featureExtractor.setN_mfcc(nMFCC);
		return featureExtractor.extractInvSTFTFeaturesAsFloatValues(stftValues, paddingFlag);
	}
	
	

	public float [] generateInvSTFTFeatures(Complex [][] stftValues, int mSampleRate, int nMFCC) {

		return this.generateInvSTFTFeatures(stftValues, mSampleRate, nMFCC, this.n_fft, this.n_mels, this.hop_length);
	}
	
	

	public Complex [][] generateSTFTFeatures(float[] magValues, int mSampleRate, int nMFCC) {

		return this.generateSTFTFeatures(magValues, mSampleRate, nMFCC, this.n_fft, this.n_mels, this.hop_length);
	}
	
	
	/**
	 * This function calculates and returns the STFT values of given Audio Sample
	 * values. STFT stands for Short Term Fourier Transform
	 * 
	 * @param magValues
	 * @param nMFCC
	 * @return
	 */
	public Complex [][] generateSTFTFeaturesWithPadOption(float[] magValues, int mSampleRate, int nMFCC, boolean padFlag) {

		return this.generateSTFTFeaturesWithPadOption(magValues, mSampleRate, nMFCC, this.n_fft, this.n_mels, this.hop_length, padFlag);
	}
	
	/**
	 * This function loads the audio file, reads its Numeric Magnitude Feature
	 * values and then takes the mean of amplitude values across all the channels and
	 * convert the signal to mono mode by taking the average. This method reads the audio file
	 * post the mentioned offset duration in seconds.
	 * 
	 * @param path
	 * @param sampleRate
	 * @param readDurationInSeconds
	 * @param offsetDuration
	 * @return
	 * @throws IOException
	 * @throws WavFileException
	 * @throws FileFormatNotSupportedException 
	 */
	
	public float[] loadAndReadWithOffset(String path, int sampleRate, int readDurationInSeconds, int offsetDuration)
			throws IOException, WavFileException, FileFormatNotSupportedException {
		float[][] magValueArray = readMagnitudeValuesFromFile(path, sampleRate, readDurationInSeconds, offsetDuration);

		int mNumFrames = this.getNoOfFrames();
		int mChannels = this.getNoOfChannels();

		// take the mean of amplitude values across all the channels and convert the
		// signal to mono mode

		float[] meanBuffer = new float[mNumFrames];
		for (int q = 0; q < mNumFrames; q++) {
			double frameVal = 0;
			for (int p = 0; p < mChannels; p++) {
				frameVal = frameVal + magValueArray[p][q];
			}
			meanBuffer[q] = (float)frameVal / mChannels;
		}
		return meanBuffer;
		
	}
	
	

	/**
	 * This function loads the audio file, reads its Numeric Magnitude Feature
	 * values and then takes the mean of amplitude values across all the channels and
	 * convert the signal to mono mode
	 * 
	 * @param path
	 * @param sampleRate
	 * @param readDurationInSeconds
	 * @return
	 * @throws IOException
	 * @throws WavFileException
	 * @throws FileFormatNotSupportedException 
	 */
	public float[] loadAndRead(String path, int sampleRate, int readDurationInSeconds)
			throws IOException, WavFileException, FileFormatNotSupportedException {

		return loadAndReadWithOffset(path, sampleRate, readDurationInSeconds, 0);
		
	}

	
	/**
	 * This function loads the audio file, reads its Numeric Magnitude Feature
	 * values and then takes the mean of amplitude values across all the channels and
	 * convert the signal to mono mode
	 * 
	 * @param path
	 * @param sampleRate
	 * @param readDurationInSeconds
	 * @return
	 * @throws IOException
	 * @throws WavFileException
	 * @throws FileFormatNotSupportedException 
	 */
	public float[][] loadAndReadStereoWithOffset(String path, int sampleRate, int readDurationInSeconds, int offsetDuration)
			throws IOException, WavFileException, FileFormatNotSupportedException {

		float[][] magValueArray = readMagnitudeValuesFromFile(path, sampleRate, readDurationInSeconds, offsetDuration);
		int mNumFrames = this.getNoOfFrames();

		float[][] stereoAudioArray = new float[2][mNumFrames];
		
		for(int i=0;i<magValueArray.length;i++) {
			stereoAudioArray[i]=Arrays.copyOfRange(magValueArray[i], 0, mNumFrames);
		}
		return stereoAudioArray;

		
	}
	
	
	
	/**
	 * This function loads the audio file, reads its Numeric Magnitude Feature
	 * values and then takes the mean of amplitude values across all the channels and
	 * convert the signal to mono mode
	 * 
	 * @param path
	 * @param sampleRate
	 * @param readDurationInSeconds
	 * @return
	 * @throws IOException
	 * @throws WavFileException
	 * @throws FileFormatNotSupportedException 
	 */
	public float[][] loadAndReadStereo(String path, int sampleRate, int readDurationInSeconds)
			throws IOException, WavFileException, FileFormatNotSupportedException {

		return loadAndReadStereoWithOffset(path, sampleRate, readDurationInSeconds, 0);

		
	}

}
