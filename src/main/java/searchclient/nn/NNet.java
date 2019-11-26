package searchclient.nn;

import org.javatuples.Pair;
import searchclient.State;

import java.io.Closeable;
import java.nio.file.Path;
import java.util.List;

public abstract class NNet implements Closeable {

    public abstract float train(Pair<List<String>, List<Double>> trainingSet);

    public abstract float predict(State state);

    public abstract void saveModel(Path fileName);

    public abstract void loadModel(Path fileName);
}
