package uk.ac.soton.ecs.arapu.run3;

import org.openimaj.data.identity.IdentifiableObject;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.PyramidSpatialAggregator;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;

/**
 * Feature Extractor built based on the model indicated by the OpenIMAJ Tutorial
 * 
 * @author Andrei Rusu
 */
public class PHOWFeatureExtractor implements FeatureExtractor<DoubleFV, IdentifiableObject<FImage>> {
	
    private PyramidDenseSIFT<FImage> pdsift;
    private HardAssigner<byte[], float[], IntFloatPair> assigner;

    public PHOWFeatureExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner) {
        this.pdsift = pdsift;
        this.assigner = assigner;
    }

    public DoubleFV extractFeature(IdentifiableObject<FImage> imgObj) {
    	
        FImage image = imgObj.data;
        pdsift.analyseImage(image);

        BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

        PyramidSpatialAggregator<byte[], SparseIntFV> spatial = 
        		new PyramidSpatialAggregator<byte[], SparseIntFV>(bovw, 2, 4);

        return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
    }
}