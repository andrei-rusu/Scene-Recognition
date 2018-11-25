package uk.ac.soton.ecs.arapu.abstracts;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.identity.IdentifiableObject;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.Annotator;

/**
 * Adapter which provides a common interface for Incremental and Batch Annotators
 */
public interface AnnotatorAdapter<T extends Annotator<IdentifiableObject<FImage>, String>> {

    public T getAnnotator();

	public void train(GroupedDataset<String, ? extends ListDataset<IdentifiableObject<FImage>>, IdentifiableObject<FImage>> training);
}
