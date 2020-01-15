package searchclient;

import searchclient.level.Level;
import searchclient.mcts.backpropagation.Backpropagation;
import searchclient.mcts.backpropagation.impl.AdditiveBackpropagation;
import searchclient.mcts.backpropagation.impl.AdditiveRAVEBackpropagation;
import searchclient.mcts.expansion.impl.AllActionsExpansion;
import searchclient.mcts.expansion.impl.AllActionsNoDuplicatesExpansion;
import searchclient.mcts.model.Node;
import searchclient.mcts.search.MonteCarloTreeSearch;
import searchclient.mcts.search.impl.AlphaGo;
import searchclient.mcts.search.impl.Basic;
import searchclient.mcts.search.impl.OneTree;
import searchclient.mcts.selection.impl.AlphaGoSelection;
import searchclient.mcts.selection.impl.UCTSelection;
import searchclient.mcts.simulation.impl.AllPairsShortestPath;
import searchclient.mcts.simulation.impl.RandomSimulation;
import searchclient.nn.NNet;
import searchclient.nn.impl.MockNNet;
import searchclient.nn.impl.PythonNNet;
import shared.Action;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Collection;
import java.util.HashSet;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicBoolean;

public class SearchClient {
    public static State parseLevel(BufferedReader serverMessages) throws IOException {
        // We can assume that the level file is conforming to specification, since the server verifies this.
        Level level = new Level(serverMessages);

        level.parse.credentials();
        level.parse.colors();
        level.parse.determineMapDetails();
        level.initiateMapDependentArrays();
        level.parse.initialState();
        level.parse.goalState();

        return level.toState();
    }

    /**
     * Implements the Graph-Search algorithm from R&N figure 3.7.
     */
    public static Action[][] search(State initialState, Frontier frontier, HashSet<State> explored) {

        System.err.format("Starting %s.\n", frontier.getName());

        frontier.add(initialState);
        while (true) {
            if (frontier.isEmpty()) {
                return null;
            }

            State leafState = frontier.pop();

            if (leafState.isGoalState()) {
                return leafState.extractPlan();
            }

            explored.add(leafState);
            for (State s : leafState.getExpandedStates()) {
                if (!explored.contains(s) && !frontier.contains(s)) {
                    frontier.add(s);
                }
            }
        }
    }



    public static void main(String[] args) throws IOException {
        // Send client name to server.
        System.out.println("SearchClient");

        // Parse the level.
        BufferedReader serverMessages = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.US_ASCII));
        State initialState = SearchClient.parseLevel(serverMessages);

        // Select search strategy.
        Frontier frontier = null;
        MonteCarloTreeSearch monteCarloTreeSearch = null;
        NNet nNet = null;
        boolean train = false, loadCheckpoint = false, loadBest = false;
        String pythonPath = null;
        Backpropagation backpropagation = new AdditiveBackpropagation();
        if (args.length > 0) {
            if (args.length > 1) {
                for (int i = 1; i < args.length; i++) {
                    switch (args[i]) {
                        case "-train":
                            train = true;
                            break;
                        case "-checkpoint":
                            loadCheckpoint = true;
                            break;
                        case "-best":
                            loadBest = true;
                            break;
                        case "-rave":
                            backpropagation = new AdditiveRAVEBackpropagation();
                            break;
                        case  "-python":
                            pythonPath = args[i + 1];
                            break;
                    }
                }
            }
            switch (args[0].toLowerCase(Locale.ROOT)) {
                case "-bfs":
                    frontier = new FrontierBFS();
                    break;
                case "-dfs":
                    frontier = new FrontierDFS();
                    break;
                case "-astar":
                    frontier = new FrontierBestFirst(new HeuristicAStar(initialState));
                    break;
                case "-wastar":
                    int w = 5;
                    if (args.length > 1) {
                        try {
                            w = Integer.parseUnsignedInt(args[1]);
                        } catch (NumberFormatException e) {
                            System.err.println("Couldn't parse weight argument to -wastar as integer, using default.");
                        }
                    }
                    frontier = new FrontierBestFirst(new HeuristicWeightedAStar(initialState, w));
                    break;
                case "-greedy":
                    frontier = new FrontierBestFirst(new HeuristicGreedy(initialState));
                    break;
                case "-basic":
                    nNet = new MockNNet(new HeuristicAStar(initialState));
                    monteCarloTreeSearch = new Basic(new UCTSelection(0.4), new AllActionsNoDuplicatesExpansion(initialState),
                            new RandomSimulation(), new AdditiveBackpropagation(), nNet);
                    break;
                case "-onetree":
                    nNet = new MockNNet(new HeuristicAStar(initialState));
                    monteCarloTreeSearch = new OneTree(new UCTSelection(0.4), new AllActionsNoDuplicatesExpansion(initialState),
                            new AllPairsShortestPath(initialState), new AdditiveBackpropagation());
                    break;
                case "-alpha":
                    if (pythonPath != null)
                        nNet = new PythonNNet(pythonPath);
                    else
                        nNet = new PythonNNet();
                    monteCarloTreeSearch = new AlphaGo(new AlphaGoSelection(), new AllActionsExpansion(), backpropagation, nNet);
                    break;
                default:
                    frontier = new FrontierBestFirst(new HeuristicAStar(initialState));
                    System.err.println("Defaulting to astar search. Use arguments -bfs, -dfs, -astar, -wastar, or " +
                            "-greedy to set the search strategy.");
            }
        } else {
            frontier = new FrontierBestFirst(new HeuristicAStar(initialState));
            System.err.println("Defaulting to astar search. Use arguments -bfs, -dfs, -astar, -wastar, or -greedy to " +
                    "set the search strategy.");
        }

        // Search for a plan.
        Action[][] plan = null;
        long startTime = System.nanoTime();
        try {
            if (monteCarloTreeSearch == null) {
                HashSet<State> explored = new HashSet<>(65536);
                StatusThread statusThread = new StatusThread(startTime, explored, frontier);
                statusThread.start();
                try {
                    plan = SearchClient.search(initialState, frontier, explored);
                } catch (Exception e) {
                    System.err.println("Exception caught in Search");
                    e.printStackTrace(System.err);
                } finally {
                    statusThread.interrupt();
                }
            }
            else {
                //StatusThread statusThread = new StatusThread(startTime, monteCarloTreeSearch.getExpandedStates());
                //statusThread.start();
                if (loadBest && Files.exists(Coach.getBestPath(monteCarloTreeSearch, initialState.levelName))) {
                    nNet.loadModel(Coach.getBestPath(monteCarloTreeSearch, initialState.levelName));
                }
                if (train) {
                    Coach coach = new Coach(nNet, monteCarloTreeSearch);
                    coach.train(initialState, loadCheckpoint);
                }
                plan = monteCarloTreeSearch.solve(new Node(initialState));
                nNet.close();
                //statusThread.interrupt();
            }
        } catch (OutOfMemoryError ex) {
            System.err.println("Maximum memory usage exceeded.");
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        // Print plan to server.
        if (plan == null) {
            System.err.println("Unable to solve level.");
            System.exit(0);
        } else {
            System.err.format("Found solution of length %d.\n", plan.length);

            for (Action[] jointAction : plan) {
                System.out.print(jointAction[0].getName());
                for (int action = 1; action < jointAction.length; ++action) {
                    System.out.print(";");
                    System.out.print(jointAction[action].getName());
                }
                System.out.println();
                // We must read the server's response to not fill up the stdin buffer and block the server.
                serverMessages.readLine();
            }
        }
    }

    private static class StatusThread implements Runnable {
        private static final long SECONDS_BETWEEN_PRINTS = 2;

        private long startTime;
        private Collection<?> explored;
        private Frontier frontier;

        private AtomicBoolean running = new AtomicBoolean(false);
        private Thread worker;

        StatusThread(long startTime, Collection<?> explored, Frontier frontier) {
            this.startTime = startTime;
            this.explored = explored;
            this.frontier = frontier;
        }

        StatusThread(long startTime, Collection<?> explored) {
            this.startTime = startTime;
            this.explored = explored;
        }

        void start() {
            this.worker = new Thread(this);
            worker.start();
        }

        void interrupt() {
            running.set(false);
            worker.interrupt();
            this.printSearchStatus();
        }

        @Override
        public void run() {
            this.running.set(true);
            while (this.running.get()) {
                try {
                    Thread.sleep(SECONDS_BETWEEN_PRINTS * 1000);
                } catch (InterruptedException ignored) {
                    Thread.currentThread().interrupt();
                    break;
                }
                this.printSearchStatus();
            }
        }

        private void printSearchStatus() {
            if (this.frontier != null)
                this.printSearchStatusFrontier();
            else
                this.printSearchStatusNoFrontier();
            System.out.println("#memory " + Memory.used());
        }

        private void printSearchStatusFrontier() {
            String statusTemplate = "#Explored: %,8d, #Frontier: %,8d, #Generated: %,8d, Time: %3.3f s\n%s\n";
            double elapsedTime = (System.nanoTime() - startTime) / 1_000_000_000d;
            System.err.format(statusTemplate, explored.size(), frontier.size(), explored.size() + frontier.size(),
                    elapsedTime, Memory.stringRep());
        }

        private void printSearchStatusNoFrontier() {
            String statusTemplate = "#Explored: %,8d, Time: %3.3f s\n%s\n";
            double elapsedTime = (System.nanoTime() - startTime) / 1_000_000_000d;
            System.err.format(statusTemplate, explored.size(), elapsedTime, Memory.stringRep());
        }
    }
}
