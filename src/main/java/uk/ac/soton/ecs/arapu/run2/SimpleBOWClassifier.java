package uk.ac.soton.ecs.arapu.run2;

import org.openimaj.data.identity.IdentifiableObject;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;

import uk.ac.soton.ecs.arapu.abstracts.AbstractClassifier;
import uk.ac.soton.ecs.arapu.abstracts.AnnotatorAdapter;

public class SimpleBOWClassifier extends AbstractClassifier<LiblinearAnnotator<IdentifiableObject<FImage>, String>> {

    private final int patchSize;
    private final int patchDensity;
    private final int kForClustering;

    public SimpleBOWClassifier(int patchSize, int patchDensity, int kForClustering) {
    	
        this.patchSize = patchSize;
        this.patchDensity = patchDensity;
        this.kForClustering = kForClustering;
    }

	@Override
	public AnnotatorAdapter<LiblinearAnnotator<IdentifiableObject<FImage>, String>> getAnnotatorAdapter() {

		return new SimpleBOWAnnotator(patchSize, patchDensity, kForClustering);
	}

	@Override
	public boolean isParallelizeClassification() {
		// TODO Auto-generated method stub
		return true;
	}
}
