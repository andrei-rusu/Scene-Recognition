package uk.ac.soton.ecs.arapu.run2;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.identity.IdentifiableObject;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.feature.DiskCachingFeatureExtractor;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.io.IOUtils;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.DoubleCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.DoubleKMeans;
import org.openimaj.util.pair.IntDoublePair;

import de.bwaldvogel.liblinear.SolverType;
import uk.ac.soton.ecs.arapu.abstracts.AnnotatorAdapter;

public class SimpleBOWAnnotator implements AnnotatorAdapter<LiblinearAnnotator<IdentifiableObject<FImage>, String>> {
	
	private final static String MODEL_FOLDER = "models/run2/";
	private final static String featuresFileName = "features.in";
	private final static String assignerFileName = "assigner.in";
	
	private final static int assignerSampleSize = 30;
	
	private int patchSize, patchDensity, kForClustering;
	
	private LiblinearAnnotator<IdentifiableObject<FImage>, String> annotator;

	public SimpleBOWAnnotator(int patchSize, int patchDensity, int kForClustering) {
		
		this.patchSize = patchSize;
		this.patchDensity = patchDensity;
		this.kForClustering = kForClustering;
	}

	@Override
	public LiblinearAnnotator<IdentifiableObject<FImage>, String> getAnnotator() {
		
		if (annotator == null) {
            throw new IllegalStateException("This Annotator needs to be trained first. Please call the corresponding 'train' method.");
        }
		
        return annotator;
	}

	@Override
	public void train(GroupedDataset<String, ? extends ListDataset<IdentifiableObject<FImage>>, IdentifiableObject<FImage>> training) {

		if (annotator == null) {
			
			// Create (if not already created) the folder structure to hold the assigner file
			File assignerFolder = new File(MODEL_FOLDER);
			assignerFolder.mkdirs();
			
            // Get the Assigner from disk or train a new one with a sample of 'assignerSampleSize' images per group
            HardAssigner<double[], double[], IntDoublePair> assigner = 
            		getAssigner(GroupedUniformRandomisedSampler.sample(training, assignerSampleSize));
            
            FeatureExtractor<DoubleFV, IdentifiableObject<FImage>> extractor = 
					new DiskCachingFeatureExtractor<>(
							new File(MODEL_FOLDER + featuresFileName),
							new SimpleBOWFeatureExtractor(patchSize, patchDensity, assigner));
            
            // Create the adapted Annotator
            annotator = new LiblinearAnnotator<IdentifiableObject<FImage>, String>(
                    extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
            
        }
		
		annotator.train(training);
	}
	
	
	private HardAssigner<double[], double[], IntDoublePair> getAssigner(
			GroupedDataset<String, ListDataset<IdentifiableObject<FImage>>, IdentifiableObject<FImage>> groupedDataset) {
		
		// Create (if not already created) the folder structure to hold the assigner file
		File assignerFolder = new File(MODEL_FOLDER);
		assignerFolder.mkdirs();
		File assignerFile = new File(MODEL_FOLDER + assignerFileName);

		// Try to read the HardAssigner from a file; if this cannot be found then instantiate and train a new one
		HardAssigner<double[], double[], IntDoublePair> assigner = null;
		try {

			assigner = IOUtils.readFromFile(assignerFile);
			
			if (assigner == null) throw new IOException("Object could not be initialised.");

		} catch(IOException e) {

			assigner = trainQuantiser(groupedDataset);
			try {
				
				IOUtils.writeToFile(assigner, assignerFile);
				
			} catch (IOException e1) {
				
				e1.printStackTrace();
			}

		}

		return assigner;
	}
	
	
	private HardAssigner<double[], double[], IntDoublePair> trainQuantiser(
            GroupedDataset<String, ListDataset<IdentifiableObject<FImage>>, IdentifiableObject<FImage>> groupedDataset) {
		
		// This list will hold all the patches we accumulate from all FImages
        List<DoubleFV> allPatches = new ArrayList<>();

        // Iterate through all images and create a sample of patches
        for (IdentifiableObject<FImage> imgObj : groupedDataset) {
        	
            List<DoubleFV> currentPatches = SimpleBOWFeatureExtractor.constructPatches(imgObj.data, patchSize, patchDensity);
            allPatches.addAll(currentPatches);
        }
        
        // Convert to double[][] so we can use DoubleKMeans to cluster
        // We tried using FeatureVectorKMeans but there are some problems with the IO operations when retrieving the Assigner!
        double[][] sample = allPatches.stream().map(x -> x.asDoubleVector()).toArray(double[][]::new);

        // Create 'kForClustering' number of clusters
        DoubleKMeans km = DoubleKMeans.createExact(kForClustering);
        DoubleCentroidsResult result = km.cluster(sample);

        return result.defaultHardAssigner();
    }

}
