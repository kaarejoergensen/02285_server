package nn;

import searchclient.State;

import java.io.Closeable;
import java.nio.file.Path;
import java.util.List;

public abstract class NNet implements Closeable {

    public abstract void train(List<String[]> examples);

    public abstract float predict(State state);

    public abstract void saveModel(Path fileName);

    public abstract void loadModel(Path fileName);
}
