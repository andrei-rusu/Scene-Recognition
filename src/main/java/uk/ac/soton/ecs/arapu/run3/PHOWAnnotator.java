package uk.ac.soton.ecs.arapu.run3;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.identity.IdentifiableObject;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.feature.DiskCachingFeatureExtractor;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.io.IOUtils;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;
import uk.ac.soton.ecs.arapu.abstracts.AnnotatorAdapter;

public class PHOWAnnotator implements AnnotatorAdapter<LiblinearAnnotator<IdentifiableObject<FImage>,String>> {
	
	private final static String MODEL_FOLDER = "models/run3/";
	private final static String featuresFileName = "features.in";
	private final static String assignerFileName = "assigner.in";
	
	private final static int assignerSampleSize = 30;
	private final static int assignerMaxKeepFeatures = 10000;
	
	private LiblinearAnnotator<IdentifiableObject<FImage>,String> annotator;

	private final HomogeneousKernelMap kernelMap;
	private final PyramidDenseSIFT<FImage> pdsift;
	private final int kForClustering;
	
	public PHOWAnnotator(HomogeneousKernelMap kernelMap, PyramidDenseSIFT<FImage> pdsift, int kForClustering) {
		
		this.kernelMap = kernelMap;
		this.pdsift = pdsift;
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
	public void train(
			GroupedDataset<String, ? extends ListDataset<IdentifiableObject<FImage>>, IdentifiableObject<FImage>> training) {

		if (annotator == null) {
			
			// Create (if not already created) the folder structure to hold the assigner file
			File assignerFolder = new File(MODEL_FOLDER);
			assignerFolder.mkdirs();
			
			// Get the Assigner from disk or train a new one with a sample of 'assignerSampleSize' images per group
            HardAssigner<byte[], float[], IntFloatPair> assigner = 
            		getAssigner(GroupedUniformRandomisedSampler.sample(training, assignerSampleSize));
            
            FeatureExtractor<DoubleFV, IdentifiableObject<FImage>> extractor = 
					new DiskCachingFeatureExtractor<>(
							new File(MODEL_FOLDER + featuresFileName),
							kernelMap.createWrappedExtractor(new PHOWFeatureExtractor(pdsift, assigner)));

//			// Create a wrapped extractor with a HomogeneousKernelMap and PyramidDenseSIFT features
//			FeatureExtractor<DoubleFV, IdentifiableObject<FImage>> extractor = 
//					kernelMap.createWrappedExtractor(new PHOWFeatureExtractor(pdsift, assigner));
			
			// Create the annotator
			annotator = new LiblinearAnnotator<>(
					extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		}
		
		annotator.train(training);

	}
	
	private HardAssigner<byte[], float[], IntFloatPair> getAssigner(
			GroupedDataset<String, ListDataset<IdentifiableObject<FImage>>, IdentifiableObject<FImage>> groupedDataset) {
		
		File assignerFile = new File(MODEL_FOLDER + assignerFileName);

		// Try to read the HardAssigner from a file; if this cannot be found then instantiate and train a new one
		HardAssigner<byte[], float[], IntFloatPair> assigner = null;
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
	
	
	private HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(
            GroupedDataset<String, ListDataset<IdentifiableObject<FImage>>, IdentifiableObject<FImage>> groupedDataset) {
		
		List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();

		for (IdentifiableObject<FImage> imgObj : groupedDataset) {

			// Analyze the image using the pdsift object
			pdsift.analyseImage(imgObj.data);
			// Add the keypoints to the list of allkeys
			allkeys.add(pdsift.getByteKeypoints(0.005f));
		}

		if (allkeys.size() > assignerMaxKeepFeatures)
			allkeys = allkeys.subList(0, assignerMaxKeepFeatures);

		// Same as in the tutorial, the next lines follow the KMeans clustering process with 
		ByteKMeans km = ByteKMeans.createKDTreeEnsemble(kForClustering);
		DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
		ByteCentroidsResult result = km.cluster(datasource);

		return result.defaultHardAssigner();
    }

	
	
}
