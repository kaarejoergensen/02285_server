package searchclient.nn.impl;

import lombok.AllArgsConstructor;
import searchclient.Heuristic;
import searchclient.State;
import searchclient.nn.NNet;
import searchclient.nn.PredictResult;
import shared.Action;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@AllArgsConstructor
public class MockNNet extends NNet {
    private Heuristic heuristic;

    @Override
    public float train(List<String> trainingSet) {
        return 0;
    }

    @Override
    public PredictResult predict(State state) {
        Map<Action, Integer> scores = new HashMap<>();
        int worstAction = -1;
        for (State n : state.getExpandedStates()) {
            int score = this.heuristic.f(n);
            scores.put(n.jointAction[0], score);
            if (score > worstAction) worstAction = score;
        }
        int sum = 0;
        for (Map.Entry<Action, Integer> entry : scores.entrySet()) {
            int score = worstAction - entry.getValue();
            scores.put(entry.getKey(), score);
            sum += score;
        }
        List<Action> allActions = Action.getAllActions();
        double[] probabilityVector = new double[allActions.size()];
        for (int i = 0; i < allActions.size(); i++) {
            if (scores.containsKey(allActions.get(i))) {
                probabilityVector[i] = 1.0 / sum * scores.get(allActions.get(i));
            }
        }
        float score = state.isGoalState() ? 1f : 0;
        return new PredictResult(probabilityVector, score);
    }

    @Override
    public void saveModel(Path fileName) {
    }

    @Override
    public void loadModel(Path fileName) {

    }

    @Override
    public NNet clone() {
        return new MockNNet(this.heuristic);
    }

    @Override
    public String toString() {
        return "MNNET";
    }

    @Override
    public void close() {
    }
}
