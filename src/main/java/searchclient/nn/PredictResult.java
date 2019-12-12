package searchclient.nn;

import lombok.Data;

@Data
public class PredictResult {
    private final double[] probabilityVector;
    private final float score;
    private final float loss;
}
