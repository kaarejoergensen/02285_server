package searchclient;

import lombok.AllArgsConstructor;
import lombok.Data;
import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarBuilder;
import me.tongfei.progressbar.ProgressBarStyle;
import org.apache.commons.lang3.mutable.MutableBoolean;
import searchclient.mcts.model.Node;
import searchclient.mcts.search.MonteCarloTreeSearch;
import searchclient.nn.NNet;
import searchclient.nn.impl.PythonNNet;
import shared.Action;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.Collectors;

@Data
@AllArgsConstructor
public class Coach {
    private static final int NUMBER_OF_EPISODES = 50;
    private static final int NUMBER_OF_TRAINING_ITERATIONS = 100;
    private static final int MAX_NUMBER_OF_TRAINING_EPISODES = 5;
    private static final int MAX_NUMBER_OF_NODES_TO_EXPLORE = 50;

    private static final String MODEL_FOLDER_NAME = "models/shared/";
    private static final String TMP_OLD_MODEL ="temp_old.pth";
    private static final String TMP_NEW_MODEL = "temp_new.pth";
    private static final String BEST_MODEL = "best.pth";
    private static final String CHECKPOINT = "checkpoint";

    private NNet nNet;
    private MonteCarloTreeSearch monteCarloTreeSearch;
    private Integer gpus;
    
    public void train(State root, boolean loadCheckpoint) throws IOException, ClassNotFoundException {
        this.train(Collections.singletonList(root), loadCheckpoint);
    }

    public void train(List<State> states, boolean loadCheckpoint) throws IOException, ClassNotFoundException {
        this.createFolders();
        Deque<List<String>> trainingExamples = new ArrayDeque<>();
        int cores = Runtime.getRuntime().availableProcessors();
        ExecutorService executorService = Executors.newFixedThreadPool(cores);
        if (loadCheckpoint && Files.exists(getCheckpointPath())) {
            trainingExamples = this.loadTrainingData();
        }
        for (int i = 0; i < NUMBER_OF_TRAINING_ITERATIONS; i++) {
            System.err.println("------------ITERATION " + (i + 1) + " ------");
            if (!loadCheckpoint || trainingExamples.isEmpty() || i != 0) {
                List<String> trainingData = this.runEpisodes(states);
                if (trainingExamples.size() >= MAX_NUMBER_OF_TRAINING_EPISODES) {
                    trainingExamples.pop();
                }
                trainingExamples.add(trainingData);
                this.saveTrainingData(trainingExamples);
            } else {
                System.err.println("Skipping first iteration due to checkpoint load");
            }

            List<String> finalTrainingData = trainingExamples.stream().flatMap(List::stream).collect(Collectors.toList());
            this.nNet.saveModel(getTmpOldPath());

            float loss = this.nNet.train(finalTrainingData);
            this.nNet.saveModel(getTmpNewPath());
            System.err.println("Training done. Loss: " + loss);

            System.err.println("Solving " + states.size() + " different levels.");
            int difference = 0;
            List<Callable<Integer>> callables = states.stream()
                    .map(s -> (Callable<Integer>) () -> pit(executorService, s))
                    .collect(Collectors.toList());
            try {
                for (Future<Integer> future : executorService.invokeAll(callables)) {
                    difference += future.get();
                }
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
                return;
            }

            if (difference >= 0) {
                System.err.println("Accepting new model. New plan is " + difference + " shorter.");
                this.nNet.loadModel(getTmpNewPath());
                this.nNet.saveModel(getBestPath());
            } else {
                System.err.println("Rejecting new model. Old plan is " + Math.abs(difference) + " shorter.");
            }
        }
        executorService.shutdown();
    }

    private int pit(ExecutorService executorService, State state) throws ExecutionException, InterruptedException {
        Callable<Action[][]> newModelCallable = () -> getActions(state, getTmpNewPath(), 0);
        Callable<Action[][]> oldModelCallable = () -> getActions(state, getTmpOldPath(), this.gpus > 1 ? 1 : 0);

        Future<Action[][]> newPlanFuture = executorService.submit(newModelCallable);
        Future<Action[][]> oldPlanFuture = executorService.submit(oldModelCallable);

        Action[][] newPlan;
        Action[][] oldPlan;
        newPlan = newPlanFuture.get();
        oldPlan = oldPlanFuture.get();
        System.err.println("Old plan: " + (oldPlan != null ? oldPlan.length : "null") +
                " New plan: " + (newPlan != null ? newPlan.length : "null"));
        if (oldPlan != null && newPlan != null) {
            return  (oldPlan.length - newPlan.length);
        } else if (oldPlan == null) {
            return 1;
        } else {
            return -1;
        }
    }

    private Action[][] getActions(State state, Path modelPath, int gpu) throws IOException {
        MonteCarloTreeSearch mcts = this.monteCarloTreeSearch.clone();
        NNet pythonNNet = new PythonNNet(((PythonNNet)this.nNet).getPythonPath(), gpu);
        pythonNNet.loadModel(modelPath);
        mcts.setNNet(pythonNNet);
        Action[][] plan = mcts.solve(new Node(state));
        pythonNNet.close();
        return plan;
    }


    private void saveTrainingData(Deque<List<String>> trainingData) throws IOException {
        FileOutputStream fos = new FileOutputStream(getCheckpointPath().toFile());
        ObjectOutputStream oos = new ObjectOutputStream(fos);
        oos.writeObject(trainingData);
        oos.close();
    }

    private Deque<List<String>> loadTrainingData() throws IOException, ClassNotFoundException {
        FileInputStream fis = new FileInputStream(getCheckpointPath().toFile());
        ObjectInputStream ois = new ObjectInputStream(fis);
        Deque<List<String>> trainingData = (Deque<List<String>>) ois.readObject();
        System.err.println("Loaded training data of size " + trainingData.size());
        ois.close();
        return trainingData;
    }

    private void createFolders() throws IOException {
        Path folderPath = Path.of(MODEL_FOLDER_NAME);
        if (!Files.isDirectory(folderPath)) {
            Files.createDirectories(folderPath);
        }
    }

    public static Path getBestPath() {
        return Path.of(MODEL_FOLDER_NAME + BEST_MODEL);
    }

    private Path getCheckpointPath() {
        return Path.of(MODEL_FOLDER_NAME + CHECKPOINT);
    }

    private Path getTmpNewPath() {
        return Path.of(MODEL_FOLDER_NAME + TMP_NEW_MODEL);
    }

    private Path getTmpOldPath() {
        return Path.of(MODEL_FOLDER_NAME + TMP_OLD_MODEL);
    }

    private List<String> runEpisodes(List<State> states) {
        System.err.println("Running episodes for " + states.size() + " states.");
        List<String> result = new ArrayList<>();
        for (State state : states) {
            result.addAll(this.runEpisodes(state, NUMBER_OF_EPISODES / states.size()));
        }
        System.err.println("All episodes done. " + result.size() + " states generated.");
        return result;
    }
    
    private List<String> runEpisodes(State root, int totalNumberOfEpisodes) {
        int cores = Runtime.getRuntime().availableProcessors();
        int numberOfThreadsPrGPU = cores;
        if (gpus != null) {
            numberOfThreadsPrGPU = cores / gpus;
        }
        if (totalNumberOfEpisodes < cores) {
            totalNumberOfEpisodes = cores;
        }
        ExecutorService executorService = Executors.newFixedThreadPool(cores);
        List<Callable<List<String>>> callableList = new ArrayList<>(cores);
        AtomicInteger numberOfEpisodes = new AtomicInteger(0);
        AtomicInteger numberOfSolvedEpisodes = new AtomicInteger(0);
        ProgressBar progressBar = new ProgressBarBuilder()
                .setPrintStream(System.err)
                .setUpdateIntervalMillis(300)
                .setInitialMax(totalNumberOfEpisodes)
                .setTaskName("Episodes")
                .setStyle(ProgressBarStyle.ASCII)
                .build();
        if (this.nNet instanceof PythonNNet) {
            try {
                ((PythonNNet) this.nNet).saveTempModel();
            } catch (IOException e) {
                e.printStackTrace();
                return Collections.emptyList();
            }
        }
        AtomicInteger gpuUsed = new AtomicInteger(0);
        AtomicInteger gpu = new AtomicInteger(0);
        for (int i = 0; i < cores; i++) {
            int finalNumberOfThreadsPrGPU = numberOfThreadsPrGPU;
            int finalTotalNumberOfEpisodes = totalNumberOfEpisodes;
            callableList.add(() -> {
                List<StateActionTakenSolvedTuple> finalList = new ArrayList<>();
                MonteCarloTreeSearch mcts = this.monteCarloTreeSearch.clone();
                PythonNNet newNNet;
                if (gpus != null) {
                    synchronized (this) {
                        if (gpuUsed.get() >= finalNumberOfThreadsPrGPU) {
                            gpu.incrementAndGet();
                            gpuUsed.set(0);
                        }
                        newNNet = new PythonNNet(((PythonNNet)this.nNet).getPythonPath(), gpu.intValue());
                        gpuUsed.incrementAndGet();
                    }
                } else {
                    newNNet = (PythonNNet) this.nNet.clone();
                }
                newNNet.loadTempModel();
                mcts.setNNet(newNNet);
                while (numberOfEpisodes.getAndIncrement() < finalTotalNumberOfEpisodes) {
                    mcts = mcts.clone();
                    Node node = new Node(root);
                    MutableBoolean solved = new MutableBoolean(false);
                    List<StateActionTakenSolvedTuple> stateActionTakenSolvedTuples = new ArrayList<>();
                    for (int k = 0; !solved.booleanValue() && k < MAX_NUMBER_OF_NODES_TO_EXPLORE; k++) {
                        mcts.runMCTS(node);
                        stateActionTakenSolvedTuples.addAll(allPermutations(node, solved));
                        node = node.getChildStochastic(true);
                        solved.setValue(node.getState().isGoalState());
                    }
                    if (solved.booleanValue()) numberOfSolvedEpisodes.getAndIncrement();
                    finalList.addAll(stateActionTakenSolvedTuples);
                    synchronized (this) {
                        progressBar.step();
                        progressBar.setExtraMessage(numberOfSolvedEpisodes.intValue() + "/"
                                + finalTotalNumberOfEpisodes + " solved.");
                    }
                }
                newNNet.close();
                return finalList.stream().map(StateActionTakenSolvedTuple::toString).collect(Collectors.toList());
            });
        }
        List<String> stateActionTakenSolvedTuples = new ArrayList<>();
        try {
            List<Future<List<String>>> futures = executorService.invokeAll(callableList);
            for (var future : futures) {
                stateActionTakenSolvedTuples.addAll(future.get());
            }
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
        executorService.shutdown();
        progressBar.close();
        System.err.println("Episodes done. Solution found in  " + numberOfSolvedEpisodes.intValue() + "/" + totalNumberOfEpisodes);
        return stateActionTakenSolvedTuples;
    }

    public List<StateActionTakenSolvedTuple> allPermutations(Node node, MutableBoolean solved) {
        List<StateActionTakenSolvedTuple> result = new ArrayList<>();

        List<String> mlStates = node.getState().getAllMLTranspositions();

        var map = node.getNumberOfTimesActionTakenMap();

        result.add(new StateActionTakenSolvedTuple(mlStates.get(0), map, solved));
        result.add(new StateActionTakenSolvedTuple(mlStates.get(1), this.transposeActionTakenMapVertical(map), solved));
        result.add(new StateActionTakenSolvedTuple(mlStates.get(2), this.transposeActionTakenMapHorizontal(map), solved));
        result.add(new StateActionTakenSolvedTuple(mlStates.get(3), this.transposeActionTakenMapBoth(map), solved));

        return result;
    }

    private Map<Action, Integer> transposeActionTakenMapVertical(Map<Action, Integer> actionIntegerMap) {
        return transposeActionTakenMap(actionIntegerMap, Action::transposeVertical);
    }

    private Map<Action, Integer> transposeActionTakenMapHorizontal(Map<Action, Integer> actionIntegerMap) {
        return transposeActionTakenMap(actionIntegerMap, Action::transposeHorizontal);
    }

    private Map<Action, Integer> transposeActionTakenMapBoth(Map<Action, Integer> actionIntegerMap) {
        return transposeActionTakenMap(actionIntegerMap, Action::transposeBoth);
    }

    private Map<Action, Integer> transposeActionTakenMap(Map<Action, Integer> actionIntegerMap, Function<Action, Action> actionPermutation) {
        Map<Action, Integer> map = new HashMap<>();
        for (Map.Entry<Action, Integer> entry : actionIntegerMap.entrySet()) {
            map.put(actionPermutation.apply(entry.getKey()), entry.getValue());
        }
        return map;
    }

    @Data
    private static class StateActionTakenSolvedTuple {
        private final String state;
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
            return this.state + "|" +
                    Arrays.toString(probabilityVector) + "|" +
                    (this.solved.booleanValue() ? "1" : "-1");
        }
    }
}
