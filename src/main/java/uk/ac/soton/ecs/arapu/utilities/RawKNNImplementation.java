package uk.ac.soton.ecs.arapu.utilities;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.validation.ValidationData;
import org.openimaj.experiment.validation.cross.GroupedKFold;
import org.openimaj.feature.DoubleFV;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.knn.DoubleNearestNeighbours;
import org.openimaj.knn.DoubleNearestNeighboursExact;
import org.openimaj.util.pair.IntDoublePair;

/**
 * Alternative implementation for KNN Annotation using the raw FloatNearestNeighboursExact
 * 
 * As compared to the KNNAnnotator implementation, this class is also able to perform KFold for choosing the best k
 * 
 * @author Andrei Rusu
 *
 */
public final class RawKNNImplementation {
	
	private final static int CROP_SIZE = 16;

	public static void solve(int k) {
		
		try (PrintWriter writer = new PrintWriter("predicted/run1.txt")) {
			
			GroupedDataset<String, VFSListDataset<FImage>, FImage> train = DataLoader.loadTrainData();
			ListDataset<FImage> test = DataLoader.loadTestData();
			int trainSize = Utils.getDatasetSize(train);
			int testSize = test.size();
			
			// This matrix will hold all feature vectors from each training image
			double[][] fvs = new double[trainSize][CROP_SIZE * CROP_SIZE];
			
			// Labels for each feature vector
			String[] labels = new String[trainSize];
			
			int i = 0;
			for (Map.Entry<String, VFSListDataset<FImage>> entry : train.entrySet()) {
				
				String label = entry.getKey();
				

				for (FImage img : entry.getValue()) {
					
					fvs[i] = createFeatureVector(img);
					labels[i] = label;
					++i;
				}
			}
			
			// Construct the KNN object
			DoubleNearestNeighbours knn = new DoubleNearestNeighboursExact(fvs);
			
			for (int j = 0; j < testSize; j++) {
				
				double[] query = createFeatureVector(test.get(j));
				List<IntDoublePair> neighbours = knn.searchKNN(query, k);
				
				/*
				 * Iterating through all the (index, distance) pairs.
				 * For each index, the corresponding label is retrieved. The label with most occurances is our prediction.
				 */
				Map<String, Integer> occurances = new HashMap<>();
				for (IntDoublePair neighbour : neighbours) {
					
					String neighbourLabel = labels[neighbour.getFirst()];
					occurances.put(neighbourLabel, occurances.getOrDefault(neighbourLabel, 0) + 1);
				}
				
				String predicted = Collections.max(occurances.entrySet(), Map.Entry.comparingByValue()).getKey();
				
				// pics 1314 2938 2962 are missing in the dataset
				int idToWrite = j;
				if (j >= 1314) {
					++idToWrite;
				}
				if (j >= 2938) {
					++idToWrite;
				}
				if (j >= 2962) {
					++idToWrite;
				}
				
				writer.println(idToWrite + ".jpg " + predicted);
				
			}
			
			
		} catch (IOException e) {
			e.printStackTrace();
		}

	}
	
	// This method performs KFold cross-validation and attempts to find the best K
	// Note that this method splits the data into train and test, and only after that KFold is performed on the new training set
	// It also prints the accuracy for the best CV split
	public static void trainThenTest() {
		
		try {
			
			GroupedRandomSplitter<String, FImage> splits = DataLoader.getSplittedTrainData(80, 0, 20);
			
			GroupedDataset<String, ListDataset<FImage>, FImage> train = splits.getTrainingDataset();
			GroupedDataset<String, ListDataset<FImage>, FImage> test = splits.getTestDataset();
			
			int noOfSplits = 10;
			GroupedKFold<String, FImage> kFoldObj = new GroupedKFold<>(noOfSplits);
			Iterator<ValidationData<GroupedDataset<String, ListDataset<FImage>, FImage>>> iter = kFoldObj.createIterable(train).iterator();
			
			
			// The value for k (NN) will be between 1-20
			int maxK = 20;
			
			int bestK = 0;
			float bestCVAcc = 0;
			float bestTstAcc = 0;
			
			while (iter.hasNext()) {
				
				ValidationData<GroupedDataset<String, ListDataset<FImage>, FImage>> validData = iter.next();
				
				GroupedDataset<String, ListDataset<FImage>, FImage> curTrain = validData.getTrainingDataset();
				GroupedDataset<String, ListDataset<FImage>, FImage> curCv = validData.getValidationDataset();
				int curTrainSize = Utils.getDatasetSize(curTrain);
				
				// This list will hold all feature vectors from each training image
				double[][] fvs = new double[curTrainSize][CROP_SIZE * CROP_SIZE];
				
				// Labels for each feature vector
				String[] labels = new String[curTrainSize];
				
				int i = 0;
				for (Map.Entry<String, ListDataset<FImage>> entry : curTrain.entrySet()) {
					
					String label = entry.getKey();
					
					for (FImage img : entry.getValue()) {
						fvs[i] = createFeatureVector(img);
						labels[i] = label;
						i++;
					}
				}
				
				for (int k = 1; k <= maxK; k++) {
				
					float acc = validateAgainst(curCv, fvs, labels, k);

					if (acc > bestCVAcc) {
						
						bestK = k;
						bestCVAcc = acc;
						// We also validate the new k against the test set, capturing the accuracy we would get by using this value
						bestTstAcc = validateAgainst(test, fvs, labels, bestK);

					}
				}
				
			}

			
			System.out.println("\nBest K: " + bestK);
			System.out.println("Best Pred against CV: " + bestCVAcc);
			System.out.println("Pred against Test: " + bestTstAcc);
			
		} catch (IOException e) {	
			e.printStackTrace();
		}
	}
	
	// This method performs KFold cross-validation and attempts to find the best K
	// Note that this method performs KFold directly on the whole training set (no test split happens)
	// It also prints the accuracy for the best CV split
	public static void trainNoTest() {
		
		try {
			
			GroupedDataset<String, ListDataset<FImage>, FImage> train = DataLoader.getTrainSampleAll();
			
			int noOfSplits = 10;
			GroupedKFold<String, FImage> kFoldObj = new GroupedKFold<>(noOfSplits);
			Iterator<ValidationData<GroupedDataset<String, ListDataset<FImage>, FImage>>> iter = 
					kFoldObj.createIterable(train).iterator();
			
			
			// The value for k (NN) will be between 1-20
			int maxK = 20;
			
			int bestK = 0;
			float bestCVAcc = 0;
			
			while (iter.hasNext()) {
				
				ValidationData<GroupedDataset<String, ListDataset<FImage>, FImage>> validData = iter.next();
				
				GroupedDataset<String, ListDataset<FImage>, FImage> curTrain = validData.getTrainingDataset();
				GroupedDataset<String, ListDataset<FImage>, FImage> curCv = validData.getValidationDataset();
				int curTrainSize = Utils.getDatasetSize(curTrain);
				
				// This list will hold all feature vectors from each training image
				double[][] fvs = new double[curTrainSize][CROP_SIZE * CROP_SIZE];
				
				// Labels for each feature vector
				String[] labels = new String[curTrainSize];
				
				int i = 0;
				for (Map.Entry<String, ListDataset<FImage>> entry : curTrain.entrySet()) {
					
					String label = entry.getKey();
					
					for (FImage img : entry.getValue()) {
						fvs[i] = createFeatureVector(img);
						labels[i] = label;
						i++;
					}
				}
				
				for (int k = 1; k <= maxK; k++) {
				
					float acc = validateAgainst(curCv, fvs, labels, k);

					if (acc > bestCVAcc) {
						
						bestK = k;
						bestCVAcc = acc;

					}
				}
				
			}

			
			System.out.println("\nBest K: " + bestK);
			System.out.println("Best Pred against CV: " + bestCVAcc);
			
		} catch (IOException e) {	
			e.printStackTrace();
		}
	}
	
	
	public static double[] createFeatureVector(FImage img) {
		
		// Extract square about the center
        int minDim = Math.min(img.width, img.height);
        img = img.extractCenter(minDim, minDim);
        
        // Resize image and normalize values
        img = ResizeProcessor.resample(img, CROP_SIZE, CROP_SIZE).normalise();
        
        // Center the mean of all pixels in 0
        Utils.meanCenterInplace(img);
        
        // Create Feature Vector and normalize it to have unit length (Lp2 norm)
        DoubleFV fv = new DoubleFV(img.getDoublePixelVector()).normaliseFV(2);

		// Return raw double[]
		return fv.getVector();
	}
	
	
	// Returns the accuracy of KNN against "data", with a specific traindata (represented as feature vectors and labels), and a specific k
	public static float validateAgainst(GroupedDataset<String, ListDataset<FImage>, FImage> data, double[][] fvs, String[] labels,  int k) {
		
		// Dataset size
		int size = Utils.getDatasetSize(data);
		
		// Construct the KNN object
		DoubleNearestNeighbours knn = new DoubleNearestNeighboursExact(fvs);
		
		int correctlyPredicted = 0;
		
		for (Map.Entry<String, ListDataset<FImage>> entry : data.entrySet()) {
			
			String label = entry.getKey();
			
			for (FImage img : entry.getValue()) {
				
				double[] query = createFeatureVector(img);
				List<IntDoublePair> neighbours = knn.searchKNN(query, k);
				
				/*
				 * Iterating through all the (index, distance) pairs.
				 * For each index, the corresponding label is retrieved. The label with most occurances is our prediction.
				 */
				Map<String, Integer> occurances = new HashMap<>();
				for (IntDoublePair neighbour : neighbours) {
					
					String neighbourLabel = labels[neighbour.getFirst()];
					occurances.put(neighbourLabel, occurances.getOrDefault(neighbourLabel, 0) + 1);
				}
				
				String predicted = Collections.max(occurances.entrySet(), Map.Entry.comparingByValue()).getKey();
				
				if (predicted.equals(label)) 
					correctlyPredicted++;
				
			}
		}
		
		return (float)correctlyPredicted / size * 100;
	}
	
}
