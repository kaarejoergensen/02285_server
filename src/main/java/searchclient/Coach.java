package searchclient;

import lombok.AllArgsConstructor;
import lombok.Data;
import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarBuilder;
import org.apache.commons.lang3.mutable.MutableBoolean;
import searchclient.mcts.model.Node;
import searchclient.mcts.search.MonteCarloTreeSearch;
import searchclient.nn.NNet;
import searchclient.nn.Trainer;
import shared.Action;

import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

@Data
@AllArgsConstructor
public class Coach implements Trainer {
    private static final int NUMBER_OF_EPISODES = 50;
    private static final int NUMBER_OF_TRAINING_ITERATIONS = 100;
    private static final int MAX_NUMBER_OF_TRAINING_EPISODES = 20;
    private static final int MAX_NUMBER_OF_NODES_TO_EXPLORE = 10;

    private NNet nNet;
    private MonteCarloTreeSearch monteCarloTreeSearch;

    @Override
    public void train(State root) {
        Deque<List<String>> trainingExamples = new ArrayDeque<>();
        for (int i = 0; i < NUMBER_OF_TRAINING_ITERATIONS; i++) {
            System.err.println("------------ITERATION " + (i + 1) + " ------");
            List<String> trainingData = this.runEpisodes(root);
            if (trainingExamples.size() >= MAX_NUMBER_OF_TRAINING_EPISODES) {
                trainingExamples.pop();
            }
            trainingExamples.add(trainingData);

            List<String> finalTrainingData = trainingExamples.stream().flatMap(List::stream).collect(Collectors.toList());
            Collections.shuffle(finalTrainingData);

            Path tempPath = Path.of("temp.pth.tar");
            this.nNet.saveModel(tempPath);
            NNet oldNNet = this.nNet.clone();
            oldNNet.loadModel(tempPath);

            float loss = this.nNet.train(finalTrainingData);
            System.err.println("Training done. Loss: " + loss);

            MonteCarloTreeSearch newModelMCTS = this.monteCarloTreeSearch.clone();
            newModelMCTS.setNNet(this.nNet);
            MonteCarloTreeSearch oldModelMCTS = this.monteCarloTreeSearch.clone();
            oldModelMCTS.setNNet(oldNNet);

            Action[][] oldPlan = oldModelMCTS.solve(new Node(root));
            Action[][] newPlan = newModelMCTS.solve(new Node(root));
            if (newPlan.length >= oldPlan.length) {
                this.nNet.saveModel(Path.of("best.pth.tar"));
            } else {
                this.nNet.loadModel(tempPath);
            }
        }
    }

    private List<String> runEpisodes(State root) {
        int cores = Runtime.getRuntime().availableProcessors();
        ExecutorService executorService = Executors.newFixedThreadPool(cores);
        List<Callable<List<StateActionTakenSolvedTuple>>> callableList = new ArrayList<>(cores);
        AtomicInteger numberOfEpisodes = new AtomicInteger(0);
        ProgressBar progressBar = new ProgressBarBuilder()
                .setPrintStream(System.err)
                .setUpdateIntervalMillis(300)
                .setInitialMax(NUMBER_OF_EPISODES)
                .setTaskName("Episodes")
                .build();
        for (int i = 0; i < cores; i++) {
            callableList.add(() -> {
                List<StateActionTakenSolvedTuple> finalList = new ArrayList<>();
                while (numberOfEpisodes.getAndIncrement() < NUMBER_OF_EPISODES) {
                    MonteCarloTreeSearch mcts = this.monteCarloTreeSearch.clone();
                    Node node = new Node(root);
                    MutableBoolean solved = new MutableBoolean(false);
                    List<StateActionTakenSolvedTuple> stateActionTakenSolvedTuples = new ArrayList<>();
                    for (int k = 0; !solved.booleanValue() && k < MAX_NUMBER_OF_NODES_TO_EXPLORE; k++) {
                        mcts.runMCTS(node);
                        stateActionTakenSolvedTuples.add(new StateActionTakenSolvedTuple(node.getState(), node.getNumberOfTimesActionTakenMap(), solved));
                        node = node.getChildStochastic(true);
                        solved.setValue(node.getState().isGoalState());
                    }
                    finalList.addAll(stateActionTakenSolvedTuples);
                    synchronized (this) {
                        progressBar.step();
                    }
                }
                return finalList;
            });
        }
        List<StateActionTakenSolvedTuple> stateActionTakenSolvedTuples = new ArrayList<>();
        try {
            List<Future<List<StateActionTakenSolvedTuple>>> futures = executorService.invokeAll(callableList);
            for (var future : futures) {
                stateActionTakenSolvedTuples.addAll(future.get());
            }
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
        progressBar.close();
        return stateActionTakenSolvedTuples.stream().map(StateActionTakenSolvedTuple::toString).collect(Collectors.toList());
    }

    @Data
    private static class StateActionTakenSolvedTuple {
        private final State state;
        private final Map<Action, Integer> actionTakenMap;
        private final MutableBoolean solved;

        private static List<Action> allActions = Action.getAllActions();

        public String toString() {
            int totalNumberOfActionsTaken = this.actionTakenMap.values().stream().reduce(0, Integer::sum);
            double[] probabilityVector = new double[allActions.size()];
            for (int i = 0; i < probabilityVector.length; i++) {
                Integer numberOfTimesTaken = this.actionTakenMap.get(allActions.get(i));
                probabilityVector[i] = numberOfTimesTaken != null ? (float) numberOfTimesTaken / (float) totalNumberOfActionsTaken : 0f;
            }
            return this.state.toMLString() + "|" +
                    Arrays.toString(probabilityVector) + "|" +
                    (this.solved.booleanValue() ? "1" : "-1");
        }
    }
}
