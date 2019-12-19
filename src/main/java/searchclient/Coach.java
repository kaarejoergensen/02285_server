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

import java.io.*;
import java.nio.file.Files;
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
    private static final int MAX_NUMBER_OF_NODES_TO_EXPLORE = 50;

    private static final Path TMP_OLD_MODEL_PATH = Path.of("models/temp_old.pth");
    private static final Path TMP_NEW_MODEL_PATH = Path.of("models/temp_new.pth");
    public static final Path BEST_MODEL_PATH = Path.of("models/best.pth");
    private static final Path CHECKPOINT_PATH = Path.of("models/checkpoint");

    private NNet nNet;
    private MonteCarloTreeSearch monteCarloTreeSearch;

    @Override
    public void train(State root, boolean loadCheckpoint) {
        Deque<List<String>> trainingExamples = new ArrayDeque<>();
        if (loadCheckpoint && Files.exists(CHECKPOINT_PATH)) {
            try {
                trainingExamples = this.loadTrainingData();
            } catch (IOException | ClassNotFoundException e) {
                e.printStackTrace();
            }
        }
        for (int i = 0; i < NUMBER_OF_TRAINING_ITERATIONS; i++) {
            System.err.println("------------ITERATION " + (i + 1) + " ------");
            List<String> trainingData = this.runEpisodes(root);
            if (trainingExamples.size() >= MAX_NUMBER_OF_TRAINING_EPISODES) {
                trainingExamples.pop();
            }
            trainingExamples.add(trainingData);

            List<String> finalTrainingData = trainingExamples.stream().flatMap(List::stream).collect(Collectors.toList());
            this.nNet.saveModel(TMP_OLD_MODEL_PATH);

            float loss = this.nNet.train(finalTrainingData);
            this.nNet.saveModel(TMP_NEW_MODEL_PATH);
            System.err.println("Training done. Loss: " + loss);

            System.err.println("Pitting old NN vs new");
            MonteCarloTreeSearch newModelMCTS = this.monteCarloTreeSearch.clone();
            newModelMCTS.setNNet(this.nNet);
            Action[][] newPlan = newModelMCTS.solve(new Node(root));

            this.nNet.loadModel(TMP_OLD_MODEL_PATH);
            MonteCarloTreeSearch oldModelMCTS = this.monteCarloTreeSearch.clone();
            oldModelMCTS.setNNet(this.nNet);
            Action[][] oldPlan = oldModelMCTS.solve(new Node(root));

//            ExecutorService executorService = Executors.newFixedThreadPool(2);

//            Future<Action[][]> futureOldPlan = executorService.submit(() -> oldModelMCTS.solve(new Node(root)));
//            Future<Action[][]> futureNewPlan = executorService.submit(() -> newModelMCTS.solve(new Node(root)));
//            try {
                if (oldPlan == null || (newPlan != null && newPlan.length >= oldPlan.length)) {
                    System.err.println("Accepting new model. " +
                            "Old plan: " + (oldPlan != null ? oldPlan.length : "null") +
                            " New plan: " + (newPlan != null ? newPlan.length : "null"));
                    this.nNet.loadModel(TMP_NEW_MODEL_PATH);
                } else {
                    System.err.println("Rejecting new model");
                }
                this.nNet.saveModel(BEST_MODEL_PATH);
//            } catch (InterruptedException | ExecutionException e) {
//                e.printStackTrace();
//            }
            try {
                this.saveTrainingData(trainingExamples);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private void saveTrainingData(Deque<List<String>> trainingData) throws IOException {
        FileOutputStream fos = new FileOutputStream(CHECKPOINT_PATH.toFile());
        ObjectOutputStream oos = new ObjectOutputStream(fos);
        oos.writeObject(trainingData);
        oos.close();
    }

    private Deque<List<String>> loadTrainingData() throws IOException, ClassNotFoundException {
        FileInputStream fis = new FileInputStream(CHECKPOINT_PATH.toFile());
        ObjectInputStream ois = new ObjectInputStream(fis);
        Deque<List<String>> trainingData = (Deque<List<String>>) ois.readObject();
        System.err.println("Loaded training data of size " + trainingData.size());
        ois.close();
        return trainingData;
    }

    private List<String> runEpisodes(State root) {
        int cores = 1;//Runtime.getRuntime().availableProcessors();
        ExecutorService executorService = Executors.newFixedThreadPool(cores);
        List<Callable<List<StateActionTakenSolvedTuple>>> callableList = new ArrayList<>(cores);
        AtomicInteger numberOfEpisodes = new AtomicInteger(0);
        AtomicInteger numberOfSolvedEpisodes = new AtomicInteger(0);
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
                    if (solved.booleanValue()) numberOfSolvedEpisodes.getAndIncrement();
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
        System.err.println("Episodes done. Solution found in  " + numberOfSolvedEpisodes.intValue() + "/" + (numberOfEpisodes.intValue() - 1));
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
