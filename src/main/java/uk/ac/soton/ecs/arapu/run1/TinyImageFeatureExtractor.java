package uk.ac.soton.ecs.arapu.run1;

import org.openimaj.data.identity.IdentifiableObject;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;

import uk.ac.soton.ecs.arapu.utilities.Utils;

/**
 * Extracts a Tiny Image feature vector from the input image by:
 * extracing a patch by the center, resizing it to a smaller dimension, normalising and mean centering the pixel vector
 */
public class TinyImageFeatureExtractor implements FeatureExtractor<DoubleFV, IdentifiableObject<FImage>> {
	
	private final static int DEFAULT_CROP_SIZE = 16;

    private final ResizeProcessor resizeProc;
    
    public TinyImageFeatureExtractor() {
        this(new ResizeProcessor(DEFAULT_CROP_SIZE, DEFAULT_CROP_SIZE, false));
    }

    public TinyImageFeatureExtractor(ResizeProcessor resizeProc) {
        this.resizeProc = resizeProc;
    }


    @Override
    public DoubleFV extractFeature(IdentifiableObject<FImage> imgObj) {
    	
    	FImage img = imgObj.data;
    	
    	// Extract square about the center
        int minDim = Math.min(img.width, img.height);
        img = img.extractCenter(minDim, minDim);
        
        // Resize image and normalize values
        img.processInplace(resizeProc).normalise();
        
        // Center the mean of all pixels in 0
        Utils.meanCenterInplace(img);
        
        // Create Feature Vector and normalize it to have unit length (Lp2 norm)
        return new DoubleFV(img.getDoublePixelVector()).normaliseFV(2);
    }

}