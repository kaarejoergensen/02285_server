package searchclient.nn;

import searchclient.State;

import java.io.Closeable;
import java.nio.file.Path;
import java.util.List;

public abstract class NNet implements Closeable, Cloneable {

    public abstract float train(List<String> trainingSet);

    public abstract PredictResult predict(State state);

    public abstract void saveModel(Path fileName);

    public abstract void loadModel(Path fileName);

    @Override
    public abstract NNet clone();

    @Override
    public abstract String toString();
}
