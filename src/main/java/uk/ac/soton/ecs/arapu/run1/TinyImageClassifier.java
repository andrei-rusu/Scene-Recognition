package uk.ac.soton.ecs.arapu.run1;

import org.openimaj.data.identity.IdentifiableObject;
import org.openimaj.feature.DoubleFV;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.basic.KNNAnnotator;

import uk.ac.soton.ecs.arapu.abstracts.AbstractClassifier;
import uk.ac.soton.ecs.arapu.abstracts.AnnotatorAdapter;

/**
 * Scene recognition using KNN Annotator on features produced by TinyImageFeatureExtractor
 */
public class TinyImageClassifier extends AbstractClassifier<KNNAnnotator<IdentifiableObject<FImage>, String, DoubleFV>> {

    private int k;

    public TinyImageClassifier(int k) {
        this.k = k;
    }

	@Override
	public AnnotatorAdapter<KNNAnnotator<IdentifiableObject<FImage>, String, DoubleFV>> getAnnotatorAdapter() {
		
		return new TinyImageAnnotator(k);
	}

	@Override
	public boolean isParallelizeClassification() {
		// TODO Auto-generated method stub
		return true;
	}

}
