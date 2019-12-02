package searchclient;

import searchclient.level.Box;
import searchclient.level.Coordinate;
import searchclient.level.Level;
import searchclient.mcts.backpropagation.impl.AdditiveBackpropagation;
import searchclient.mcts.expansion.impl.AllActionsExpansion;
import searchclient.mcts.expansion.impl.AllActionsNoDuplicatesExpansion;
import searchclient.mcts.model.Node;
import searchclient.mcts.search.MonteCarloTreeSearch;
import searchclient.mcts.search.impl.Basic;
import searchclient.mcts.search.impl.OneTree;
import searchclient.mcts.selection.impl.UCTSelection;
import searchclient.mcts.simulation.impl.AllPairsShortestPath;
import shared.Action;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

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
        long startTime = System.nanoTime();
        List<State> splitStates = splitState(initialState);
        ExecutorService executorService = Executors.newFixedThreadPool(splitStates.size());

        List<Callable<WorkerReturn>> stateCallables = new ArrayList<>(splitStates.size());
        for (int i = 0; i < splitStates.size(); i++) {
            stateCallables.add(new Worker(i, splitStates.get(i)));
        }

        try {
            int exploredSize = 0;
            int frontierSize = 0;
            List<Action[][]> actions = new ArrayList<>();
            List<Future<WorkerReturn>> futurePlans = executorService.invokeAll(stateCallables);
            for (Future<WorkerReturn> futurePlan : futurePlans) {
                WorkerReturn workerReturn = futurePlan.get();
                exploredSize += workerReturn.getExploredSize();
                frontierSize += workerReturn.getFrontierSize();
                actions.add(workerReturn.getPlan());
                if (workerReturn.getPlan() == null) {
                    System.err.println("State could not be solved:");
                    System.err.println(workerReturn.getState());
                }
            }
            printSearchStatusFrontier(startTime, exploredSize, frontierSize);
            List<Action[][]> finalPlan = testPermutations(actions, initialState);
            if (finalPlan == null) {
                System.err.println("NOT APPLICABLE");
                return null;
            }
            var test = new Action[finalPlan.stream().map(a -> a.length).reduce(0, Integer::sum)][];
            int index = 0;
            for (Action[][] actions1 : finalPlan) {
                for (Action[] actions2 : actions1) {
                    test[index++] = actions2;
                }
            }
            return test;
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
        return null;
    }

    private static List<Action[][]> testPermutations(List<Action[][]> actions, State state) {
        return testPermutationsRecursive(actions.size(), actions, state);
    }

    private static List<Action[][]> testPermutationsRecursive(int n, List<Action[][]> actions, State state) {
        if (n == 1) {
            if (state.isApplicable(actions)) return actions;
            return null;
        } else {
            for(int i = 0; i < n-1; i++) {
                var result = testPermutationsRecursive(n - 1, actions, state);
                if (result != null) return result;
                if(n % 2 == 0) {
                    swap(actions, i, n-1);
                } else {
                    swap(actions, 0, n-1);
                }
            }
            return testPermutationsRecursive(n - 1, actions, state);
        }
    }

    private static <T> void swap(List<T> input, int a, int b) {
        T tmp = input.get(a);
        input.set(a, input.get(b));
        input.set(b, tmp);
    }


    private static void printSearchStatusFrontier(long startTime, int explored, int frontier) {
        String statusTemplate = "#Explored: %,8d, #Frontier: %,8d, #Generated: %,8d, Time: %3.3f s\n%s\n";
        double elapsedTime = (System.nanoTime() - startTime) / 1_000_000_000d;
        System.err.format(statusTemplate, explored, frontier, explored + frontier,
                elapsedTime, Memory.stringRep());
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
        if (args.length > 0) {
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
                    monteCarloTreeSearch = new Basic(new UCTSelection(0.4), new AllActionsExpansion(),
                            new AllPairsShortestPath(initialState), new AdditiveBackpropagation());
                    break;
                    case "-onetree":
                    monteCarloTreeSearch = new OneTree(new UCTSelection(0.4), new AllActionsNoDuplicatesExpansion(initialState),
                            new AllPairsShortestPath(initialState), new AdditiveBackpropagation());
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
//                StatusThread statusThread = new StatusThread(startTime, explored, frontier);
//                statusThread.start();
                plan = SearchClient.search(initialState, frontier, explored);
//                statusThread.interrupt();
            }
            else {
                //StatusThread statusThread = new StatusThread(startTime, monteCarloTreeSearch.getExpandedStates());
                //statusThread.start();
                plan = monteCarloTreeSearch.solve(new Node(initialState));
                //statusThread.interrupt();
            }
        } catch (OutOfMemoryError ex) {
            System.err.println("Maximum memory usage exceeded.");
        }

//        try {
//            long fastest = Long.MAX_VALUE;
//            double fastestConstant = 0;
//            int shortest = Integer.MAX_VALUE;
//            double shortestConstant = 0;
//            for (double constant = 0.1; constant < 10; constant = constant + 0.1) {
//                monteCarloTreeSearch = new Basic(new UCTSelection(constant), new AllActionsExpansion(), new RandomSimulation(), new AdditiveBackpropagation());
//                long startTime = System.nanoTime();
//                plan = monteCarloTreeSearch.solve(new Node(initialState));
//                long elapsedTime = System.nanoTime() - startTime;
//                if (elapsedTime < fastest || plan.length < shortest) {
//                    System.out.println("----------------------------");
//                    if (elapsedTime < fastest) {
//                        fastest = elapsedTime;
//                        fastestConstant = constant;
//                        System.out.println("NEW FASTEST: Constant: " + constant + " time: " + elapsedTime / 1_000_000_000d);
//                    }
//                    if (plan.length < shortest) {
//                        shortest = plan.length;
//                        shortestConstant = constant;
//                        System.out.println("NEW SHORTEST: Constant: " + constant + " length: " + plan.length + " time: " + elapsedTime / 1_000_000_000d);
//                    }
//                    System.out.println("Explored: " + monteCarloTreeSearch.getExpandedStates().size() + " Solution: " + plan.length);
//                    System.out.println("----------------------------");
//                }
//                //if (plan.length == shortest) System.out.println("EQUAL: " + constant);
//            }
//            System.out.println("constant: " + fastestConstant + " time: " + fastest);
//            System.out.println("DONE");
//        } catch (OutOfMemoryError e) {
//            System.err.println("Maximum memory usage exceeded.");
//        }

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

    private static List<State> splitState(State initialState) {
        Set<Map.Entry<Coordinate, Box>> boxes = initialState.boxMap.entrySet();
        List<State> splitStates = new ArrayList<>(boxes.size());

        var goalMap = generateMap(initialState.goals);

        for (Map.Entry<Coordinate, Box> boxEntry : boxes) {
            final boolean[][] walls = new boolean[initialState.walls.length][];
            for (int i = 0; i < initialState.walls.length; i++) {
                walls[i] = Arrays.copyOf(initialState.walls[i], initialState.walls[i].length);
            }
            for (Map.Entry<Coordinate, Box> boxEntry2 : initialState.boxMap.entrySet()) {
                if (!boxEntry.equals(boxEntry2)) {
                    Coordinate coordinate = boxEntry2.getKey();
                    walls[coordinate.getRow()][coordinate.getCol()] = true;
                }
            }
            Map<Coordinate, Box> boxMap = Collections.singletonMap(boxEntry.getKey(), boxEntry.getValue());
            var newGoalMap = getBestGoal(goalMap, boxEntry, initialState);
            State newState = new State(initialState.distanceMap, initialState.agentRows, initialState.agentCols,
                    initialState.agentColors, walls, boxMap, newGoalMap);
            splitStates.add(newState);
        }
        Set<Map.Entry<Coordinate, Character>> agentGoals = initialState.goals.entrySet().stream().
                filter(entry -> '0' <= entry.getValue() && entry.getValue() <= '9' ).collect(Collectors.toSet());
        if (!agentGoals.isEmpty()) {
            //TODO: Do something
        }
        return splitStates;
    }

    private static Map<Coordinate, Character> getBestGoal(Map<Character, List<Coordinate>> allGoals,
                                                          Map.Entry<Coordinate, Box> boxEntry, State state) {
        List<Coordinate> goals = allGoals.get(boxEntry.getValue().getCharacter());
        int bestDistance = Integer.MAX_VALUE;
        Coordinate bestGoal = null;
        for (Coordinate goal : goals) {
            int distance = state.distanceMap.getDistance(boxEntry.getKey(), goal);
            if (distance < bestDistance) {
                bestDistance = distance;
                bestGoal = goal;
            }
        }
        var goalMap = new HashMap<Coordinate, Character>();
        goalMap.put(bestGoal, boxEntry.getValue().getCharacter());
        goalMap.put(new Coordinate(state.agentRows[0], state.agentCols[0]), '0');
        return goalMap;
    }

    private static Map<Character, List<Coordinate>> generateMap(Map<Coordinate, Character> allGoals) {
        Map<Character, List<Coordinate>> maps = new HashMap<>();
        for (Map.Entry<Coordinate, Character> goal : allGoals.entrySet()) {
            if (maps.containsKey(goal.getValue())) {
                maps.get(goal.getValue()).add(goal.getKey());
            } else {
                List<Coordinate> list = new ArrayList<>();
                list.add(goal.getKey());
                maps.put(goal.getValue(), list);
            }
        }
        return maps;
    }

    private static class StatusThread implements Runnable {
        private static long SECONDS_BETWEEN_PRINTS = 2;

        private long startTime;
        private Collection explored;
        private Frontier frontier;

        private AtomicBoolean running = new AtomicBoolean(false);
        private Thread worker;

        StatusThread(long startTime, Collection explored, Frontier frontier) {
            this.startTime = startTime;
            this.explored = explored;
            this.frontier = frontier;
        }

        StatusThread(long startTime, Collection explored) {
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
