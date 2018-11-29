package uk.ac.soton.ecs.arapu;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.io.PrintWriter;

import org.apache.commons.io.output.TeeOutputStream;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.data.identity.IdentifiableObject;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.image.FImage;

import uk.ac.soton.ecs.arapu.abstracts.AbstractClassifier;
import uk.ac.soton.ecs.arapu.run3.PHOWClassifier;
import uk.ac.soton.ecs.arapu.utilities.DataLoader;
import uk.ac.soton.ecs.arapu.utilities.Utils;

/**
 * OpenIMAJ Coursework III!
 *
 */
@SuppressWarnings("unused")
public class App {
	
	private final static int TRAIN_NUMBER = 80;
	private final static String OUTPUT_FOLDER = "predict/";
	
	private final static int PREDICTIONS_SIZE = 2988;
	
    public static void main(String[] args) {
    	
//    	// While performing KFold multiple times, it turned out that the resulting k varied significantly around the values 4-7
//    	// Given our KFold results, we'll choose K=6 for our first run!
//    	RawKNNImplementation.trainNoTest();
//    	RawKNNImplementation.solve(6);
    	
    	try {
    	
    		// The identifiable version of loading the data is needed due to a limitation of the DiskCachingFeatureExtractor which doesn't work on raw FImages
	    	GroupedDataset<String, ListDataset<IdentifiableObject<FImage>>, IdentifiableObject<FImage>> 
	    		train = DataLoader.loadTrainDataIdentifiable();
			VFSListDataset<FImage> test = DataLoader.loadTestData();
       
            // Run 1 - Using TinyImageClassifier
//            run(train, test, new TinyImageClassifier(6), "1", true, true);
            // Run 2 - Using SimpleBoVWClassifier
//            run(train, test, new SimpleBOWClassifier(8, 4, 500), "2", true, true);
            // Run 3
            run(train, test, new PHOWClassifier(7, 5, 600), "4", true, true); 
            
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
    
    
    public static void run(
    		GroupedDataset<String, ? extends ListDataset<IdentifiableObject<FImage>>, IdentifiableObject<FImage>> train, 
    		VFSListDataset<FImage> test, AbstractClassifier<?> classifier, String runId, boolean validate, boolean classifyTest) {

    	if (validate) {
	    	String summaryFile = OUTPUT_FOLDER + "run" + runId + "_summary.txt";
	        
	        // Setup summary output to console and file
	        try (PrintStream out = new PrintStream(new TeeOutputStream(System.out, new FileOutputStream(summaryFile)))) {
	
		        out.println("\n---- Run " + runId + " ----\n");
		
		        // Evaluate the performance on the training data
		        CMResult<String> performance = classifier.evaluate(train, TRAIN_NUMBER);
		        out.println(performance.getDetailReport());
		        out.println();
	        
	        } catch (IOException e) {
	        	e.printStackTrace();
	        }
    	}
        
    	if (classifyTest) {
	        // Train the Annotator then Classify the test images
	        GroupedDataset<String, ? extends ListDataset<IdentifiableObject<FImage>>, IdentifiableObject<FImage>> classifications = 
	        		classifier.trainThenClassify(train, test);
	
	        // Write test images classifications to the corresponding files
	        writeClassifications(classifications, runId);
    	}
    }
    
    
    public static void writeClassifications(
    		GroupedDataset<String, ? extends ListDataset<IdentifiableObject<FImage>>, IdentifiableObject<FImage>> classifications, String runId) {
    	
        String outputFile = OUTPUT_FOLDER + "run" + runId + ".txt";
        
        String[] predictions = new String[PREDICTIONS_SIZE];
        classifications.entrySet().forEach(entry -> {
			String label = entry.getKey();
			
			entry.getValue().stream().forEach(idImg -> {
				
				int id = Utils.getIDFor(idImg);
				predictions[id] = label;
			});
		});
    	
    	try (PrintWriter writer = new PrintWriter(outputFile)) {
    	
    		for (int i = 0; i < predictions.length; i++) {
    			
    			 // This check is due to 3 pictures missing from 0-2987
    			if (predictions[i] != null)
    				writer.println(i + ".jpg " + predictions[i]);
    		}
    		
    	} catch (IOException e) {
    		e.printStackTrace();
    	}
    }
}
