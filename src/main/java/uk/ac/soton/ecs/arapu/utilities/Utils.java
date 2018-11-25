package uk.ac.soton.ecs.arapu.utilities;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.identity.IdentifiableObject;
import org.openimaj.image.FImage;

/**
 * Generic Utilities class used for various tasks in this coursework.
 * @author Andrei Rusu
 *
 */
public final class Utils {
	
	private Utils() {}
	
	@SuppressWarnings("unused")
	public static <K, V> int getDatasetSize(GroupedDataset<K, ? extends ListDataset<V>, V> data) {
		
		int size = 0;
		for (V obj : data) {
			++size;
		}
		return size;
	}
	
	// Gets the ID of a specific IdentifiableObject FImage
	public static int getIDFor(IdentifiableObject<FImage> idImg) {
		
		int id = Integer.valueOf(idImg.getID().replaceAll("\\D+",""));
		return id;
	}
	
	// Returns an FImage with its mean centered in 0
	public static FImage meanCenterInplace(FImage img) {
		
		float mean = img.sum() / img.getFloatPixelVector().length;
        img.subtractInplace(mean);
        return img;
	}
}
