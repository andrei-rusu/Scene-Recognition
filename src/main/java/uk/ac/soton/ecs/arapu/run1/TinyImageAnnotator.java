package uk.ac.soton.ecs.arapu.run1;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.identity.IdentifiableObject;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.basic.KNNAnnotator;

import uk.ac.soton.ecs.arapu.abstracts.AnnotatorAdapter;

public class TinyImageAnnotator implements AnnotatorAdapter<KNNAnnotator<IdentifiableObject<FImage>, String, DoubleFV>> {
	
	private KNNAnnotator<IdentifiableObject<FImage>, String, DoubleFV> annotator;
	private int k;
	
	/*
	 * Parameter k will determine how many neighbours the KNN algorithm will take into consideration
	 */
	public TinyImageAnnotator(int k) {
		
		this.k = k;
	}

	@Override
	public KNNAnnotator<IdentifiableObject<FImage>, String, DoubleFV> getAnnotator() {

		// Note that the KNNAnnotator doesn't need to be trained first, so it can be lazily initialised here.
		if (annotator == null) {
			annotator = new KNNAnnotator<>(new TinyImageFeatureExtractor(), DoubleFVComparison.EUCLIDEAN, k);
		}

		return annotator;
               
	}

	@Override
	public void train(GroupedDataset<String, ? extends ListDataset<IdentifiableObject<FImage>>, IdentifiableObject<FImage>> training) {

		getAnnotator().trainMultiClass(training);
		
	}

}
