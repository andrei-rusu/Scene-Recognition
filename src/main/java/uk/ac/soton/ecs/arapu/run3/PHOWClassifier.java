package uk.ac.soton.ecs.arapu.run3;

import org.openimaj.data.identity.IdentifiableObject;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.ml.kernel.HomogeneousKernelMap.KernelType;
import org.openimaj.ml.kernel.HomogeneousKernelMap.WindowType;

import uk.ac.soton.ecs.arapu.abstracts.AbstractClassifier;
import uk.ac.soton.ecs.arapu.abstracts.AnnotatorAdapter;

/**
 * Implementation of AbstractClassifier based on the OpenIMAJ tutorial parameters (PyramidDenseSIFT and HomogenousKernelMap)
 * 
 * @author Andrei Rusu
 *
 */
public class PHOWClassifier extends AbstractClassifier<LiblinearAnnotator<IdentifiableObject<FImage>, String>> {
	
	private final int binSize, binDensity, kForClustering;
	
	public PHOWClassifier(int binSize, int binDensity, int kForClustering) {
		
		this.binSize = binSize;
		this.binDensity = binDensity;
		this.kForClustering = kForClustering;
	}

    @Override
    public AnnotatorAdapter<LiblinearAnnotator<IdentifiableObject<FImage>, String>> getAnnotatorAdapter() {
    	
        // Create a Homogeneous Kernel Map
        HomogeneousKernelMap kernelMap = new HomogeneousKernelMap(KernelType.Chi2, WindowType.Rectangular);

        // We will extract DenseSIFT features from patches of 7x7 size, with a sample density of 3
        DenseSIFT dsift = new DenseSIFT(binDensity, binSize);

        // Wrapping the DenseSIFT in a PyramidDenseSIFT object 
        // magFactor=5f offers smoothness before extracting the features; scale levels 4,6,8,10 chosen based on the tutorial
        PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<>(dsift, 6f, 5);

        return new PHOWAnnotator(kernelMap, pdsift, kForClustering);
    }

	@Override
	public boolean isParallelizeClassification() {
		// TODO Auto-generated method stub
		return false;
	}
}
