package uk.ac.soton.ecs.arapu.utilities;

import java.io.File;
import java.io.IOException;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListBackedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.MapBackedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.data.identity.IdentifiableObject;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

public final class DataLoader {
	
	private final static String trainFolder = "training";
	private final static String testFolder = "testing";

	private DataLoader() {}
	
	// Simply loads the whole train data
	public static GroupedDataset<String, VFSListDataset<FImage>, FImage> loadTrainData() 
			throws IOException {
		
		GroupedDataset<String, VFSListDataset<FImage>, FImage> training = 
				new VFSGroupDataset<>(new File(trainFolder).getAbsolutePath(), ImageUtilities.FIMAGE_READER);
		
		return training;
	}
	
	// Wraps each FImage into an IdentifiableObject -> This is needed for DiskCachingFeatureExtractor to be able to save features for each FImage
	public static GroupedDataset<String, ListDataset<IdentifiableObject<FImage>>, IdentifiableObject<FImage>> loadTrainDataIdentifiable() 
			throws IOException {
		
		GroupedDataset<String, VFSListDataset<FImage>, FImage> 
			training = new VFSGroupDataset<>(new File(trainFolder).getAbsolutePath(), ImageUtilities.FIMAGE_READER);
		
		GroupedDataset<String, ListDataset<IdentifiableObject<FImage>>, IdentifiableObject<FImage>> 
			trainingIdentifiable = new MapBackedDataset<>();
		
		training.entrySet().forEach(entry -> trainingIdentifiable.put(entry.getKey(), createIdentifiableListDataset(entry.getValue())));
		
		return trainingIdentifiable;
	}
	
	// This gets a sample of the train data formed of "groups" number of groups
	public static GroupedDataset<String, ListDataset<FImage>, FImage> getTrainSample(int groups) 
			throws IOException {
		
		GroupedDataset<String, ListDataset<FImage>, FImage> sample = 
				GroupSampler.sample(loadTrainData(), groups, false);
		
		return sample;
	}
	
	/*
	 * Convenience method for getting the whole training set as a GroupedDataset parametrised by 
	 * <String, ListDataset, FImage> instead of <String, VFSListDataset, FImage>
	 */
	public static GroupedDataset<String, ListDataset<FImage>, FImage> getTrainSampleAll() 
			throws IOException {
		
		GroupedDataset<String, VFSListDataset<FImage>, FImage> training = loadTrainData();
		
		GroupedDataset<String, ListDataset<FImage>, FImage> wholeSample = 
				GroupSampler.sample(training, training.getGroups().size(), false);
		
		return wholeSample;
	}
	
	
	/*
	 * The following two methods split the data into training, cv and test sets
	 * 
	 * In most exercises, we will use a classical split of 
	 * 60% training, 20% cv and 20% test or
	 * 80% training, 0% cv and 20% test 
	 */
	
	public static GroupedRandomSplitter<String, FImage> getSplittedTrainData(int tr, int cv, int tst) 
			throws IOException {
		
		return new GroupedRandomSplitter<String, FImage>(loadTrainData(), tr, cv, tst);
	}
	
	public static GroupedRandomSplitter<String, FImage> getSplittedTrainSample(int groups, int tr, int cv, int tst) 
			throws IOException {
		
		return new GroupedRandomSplitter<String, FImage>(getTrainSample(groups), tr, cv, tst);
	}
	
	
	// We return the test set as a VFSListDataset as we may need to do "toIdentifiable" on it
	public static VFSListDataset<FImage> loadTestData() throws IOException {
		
		VFSListDataset<FImage> test = 
				new VFSListDataset<>(new File(testFolder).getAbsolutePath(), ImageUtilities.FIMAGE_READER);
		
		return test;
	}
	
	
	/*
	 *  Utility method for creating a ListDataset holding multiple IdentifiableObject<FImage>, which will be needed for caching features.
	 *  
	 *  The default ".toIdentifiable()" routine assigns only the basename of the files as ID, and therefore we lose the power to distinguish
	 *  between two images that are named the same, but belong to different folders.
	 */
	private static ListDataset<IdentifiableObject<FImage>> createIdentifiableListDataset(VFSListDataset<FImage> list) {
		
		ListDataset<IdentifiableObject<FImage>> result = new ListBackedDataset<>();
		
		list.toIdentifiable().stream().forEach(img -> result.add(new IdentifiableObject<FImage>(list.getID() + "/" + img.getID(), img.data)));
		
		return result;
	}
	
}
