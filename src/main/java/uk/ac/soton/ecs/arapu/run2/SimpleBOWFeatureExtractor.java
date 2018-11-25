package uk.ac.soton.ecs.arapu.run2;

import java.util.ArrayList;
import java.util.List;

import org.openimaj.data.identity.IdentifiableObject;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.array.ArrayUtils;
import org.openimaj.util.pair.IntDoublePair;

import uk.ac.soton.ecs.arapu.utilities.Utils;

public class SimpleBOWFeatureExtractor implements FeatureExtractor<DoubleFV, IdentifiableObject<FImage>> {
	
	private final HardAssigner<double[], double[], IntDoublePair> assigner;
	private int patchSize, patchDensity;
	
	public SimpleBOWFeatureExtractor(int patchSize, int patchDensity, HardAssigner<double[], double[], IntDoublePair> assigner) {
		
		this.patchSize = patchSize;
		this.patchDensity = patchDensity;
		this.assigner = assigner;
	}

	@Override
	public DoubleFV extractFeature(IdentifiableObject<FImage> img) {
		
		// Construct a list of patches
		List<DoubleFV> patches = constructPatches(img.data, patchSize, patchDensity);
		
		// Create the BagOfVisualWords object with the instance assigner
		BagOfVisualWords<double[]> bag = new BagOfVisualWords<>(assigner);
		
		// Aggregate the patches using the BagOfVisualWords object
		return bag.aggregateVectors(patches).asDoubleFV();
	}
	
	public static List<DoubleFV> constructPatches(FImage img, int patchSize, int patchDensity) {
		
		List<DoubleFV> patches = new ArrayList<>();

        for (int y = 0; y < img.getHeight() - patchSize; y += patchDensity) {
            for (int x = 0; x < img.getWidth() - patchSize; x += patchDensity) {
            	
            	// Extract a patch
                FImage patch = img.extractROI(x, y, patchSize, patchSize);

                // Mean center image
                Utils.meanCenterInplace(patch);
                
                //  Get pixel array and normalize it to have unit length (Lp2 norm)
                double[] patchPixels = ArrayUtils.normalise(patch.getDoublePixelVector());

                // Create Feature Vector
                DoubleFV fv = new DoubleFV(patchPixels);

                // Add raw feature vector to the List of patches
                patches.add(fv);
            }
        }
        
        return patches;
	}

}
