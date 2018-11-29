package uk.ac.soton.ecs.arapu.abstracts;

import java.util.Map;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListBackedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.MapBackedDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.data.identity.IdentifiableObject;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.Annotator;
import org.openimaj.util.parallel.Parallel;

/**
 * Abstract class which provides an abstraction for all the scene recognition classifiers.
 * It gives outline methods for training an annotator, classifying a test set, and evaluating the performance of the classifier on the training data.
 */
public abstract class AbstractClassifier <T extends Annotator<IdentifiableObject<FImage>, String>> {

    /*
     * This method will train an Annotator on the "training" data and then classify the images from "test"
     * 
     * Note that we return a GroupedDataset containing IdentifiableObjects! This is needed so that we can maintain the ID(filename) of each image! 
     */
    public GroupedDataset<String, ? extends ListDataset<IdentifiableObject<FImage>>, IdentifiableObject<FImage>>
            trainThenClassify(GroupedDataset<String, ? extends ListDataset<IdentifiableObject<FImage>>, IdentifiableObject<FImage>> training, 
            	VFSListDataset<FImage> test) {
    	
    	Annotator<IdentifiableObject<FImage>, String> annotator = trainAnnotator(training);

        return classify(test, annotator);
    }
    
    private Annotator<IdentifiableObject<FImage>, String> trainAnnotator(
    		GroupedDataset<String, ? extends ListDataset<IdentifiableObject<FImage>>, IdentifiableObject<FImage>> training) {
    	
    	System.out.println("Training the annotator...");
    	
    	AnnotatorAdapter<T> annotator = this.getAnnotatorAdapter();
    	annotator.train(training);
    	
    	return annotator.getAnnotator();
    }
    
    
    /*
     *  This method constructs a GroupedDataset based on the input dataset and the Annotator that shall classify each data point
     *  
     *  Note that we return a GroupedDataset containing IdentifiableObjects! This is needed so that we can maintain the ID(filename) of each image!
     */
    private GroupedDataset<String, ? extends ListDataset<IdentifiableObject<FImage>>, IdentifiableObject<FImage>> 
    		classify(VFSListDataset<FImage> data, Annotator<IdentifiableObject<FImage>, String> annotator) {
    	
    	GroupedDataset<String, ListDataset<IdentifiableObject<FImage>>, IdentifiableObject<FImage>> groupedDataset = new MapBackedDataset<>();
    	
    	ListDataset<IdentifiableObject<FImage>> idData = data.toIdentifiable();
    	if(isParallelizeClassification()) {
    		Parallel.forEach(idData, (idImg) -> {
            	
                String prediction = getMostLikelyPrediction(annotator.classify(idImg));
                
                ListDataset<IdentifiableObject<FImage>> listDataset;
                
                synchronized (groupedDataset) {
                	
                    if (groupedDataset.containsKey(prediction)) {
                    	listDataset = groupedDataset.getInstances(prediction);
                    } 
                    else {
                    	listDataset = new ListBackedDataset<>();
                    	groupedDataset.put(prediction, listDataset);
                    }
                    
                    listDataset.add(idImg);
                }     
    		});
    	}else {
    		for(IdentifiableObject<FImage> idImg :idData) {
    			String prediction = getMostLikelyPrediction(annotator.classify(idImg));
                
                ListDataset<IdentifiableObject<FImage>> listDataset;
                
                synchronized (groupedDataset) {
                	
                    if (groupedDataset.containsKey(prediction)) {
                    	listDataset = groupedDataset.getInstances(prediction);
                    } 
                    else {
                    	listDataset = new ListBackedDataset<>();
                    	groupedDataset.put(prediction, listDataset);
                    }
                    
                    listDataset.add(idImg);
                } 
    		}
    	}
    	
    	
    	
        return groupedDataset;    	
    }
    
    private String getMostLikelyPrediction(ClassificationResult<String> classificationResult) {
    	
        String prediction = null;
        double bestConfidence = Double.NEGATIVE_INFINITY;
        
        for (String curPred : classificationResult.getPredictedClasses()) {
        	
            double confidence = classificationResult.getConfidence(curPred);
            if (confidence > bestConfidence) {
            	prediction = curPred;
                bestConfidence = confidence;
            }
        }
        
        return prediction;
    }

    public CMResult<String> evaluate(GroupedDataset<String, ? extends ListDataset<IdentifiableObject<FImage>>, 
    		IdentifiableObject<FImage>> data, int trainNumber) {
    	
        System.out.println("Evaluating classification performance...");

        // Split the original data into training and validation sets
        GroupedRandomSplitter<String, IdentifiableObject<FImage>> splits = 
        		new GroupedRandomSplitter<>(data, trainNumber, 0, 100 - trainNumber);
        
        // Get the annotator
        AnnotatorAdapter<T> annotator = getAnnotatorAdapter();

        // Train with the training data obtained after the split
        annotator.train(splits.getTrainingDataset());

        // Validate the Annotator (classfier) using the validation dataset
        ClassificationEvaluator<CMResult<String>, String, IdentifiableObject<FImage>> eval
                = new ClassificationEvaluator<>(
                        annotator.getAnnotator(), splits.getTestDataset(), new CMAnalyser<IdentifiableObject<FImage>, String>(CMAnalyser.Strategy.SINGLE));

        // Get the classifications result
        Map<IdentifiableObject<FImage>, ClassificationResult<String>> classifications = eval.evaluate();
        
        // Analyze the classifications result in order to produce a CMResult object which contains a detailed report of this analysis 
        return eval.analyse(classifications);
    }
    
    
    public abstract AnnotatorAdapter<T> getAnnotatorAdapter();
    public abstract boolean isParallelizeClassification();
}