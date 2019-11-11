package searchclient;

import searchclient.level.Level;
import searchclient.mcts.MonteCarloTreeSearch;
import searchclient.mcts.backpropagation.impl.AdditiveBackpropagation;
import searchclient.mcts.expansion.impl.AllActionsExpansion;
import searchclient.mcts.impl.Basic;
import searchclient.mcts.model.Node;
import searchclient.mcts.selection.impl.UCTSelection;
import searchclient.mcts.simulation.impl.AllPairsShortestPath;
import shared.Action;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.HashSet;
import java.util.Locale;

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

        System.err.println(level);

        return level.toState();
    }

    /**
     * Implements the Graph-Search algorithm from R&N figure 3.7.
     */
    public static Action[][] search(State initialState, Frontier frontier) {
        long startTime = System.nanoTime();
        int iterations = 0;

        System.err.format("Starting %s.\n", frontier.getName());

        frontier.add(initialState);
        HashSet<State> explored = new HashSet<>(65536);

        while (true) {
            if (iterations == 10000) {
                printSearchStatus(startTime, explored, frontier);
                iterations = 0;
            }

            if (frontier.isEmpty()) {
                printSearchStatus(startTime, explored, frontier);
                return null;
            }

            State leafState = frontier.pop();

            if (leafState.isGoalState()) {
                printSearchStatus(startTime, explored, frontier);
                return leafState.extractPlan();
            }

            explored.add(leafState);
            for (State s : leafState.getExpandedStates()) {
                if (!explored.contains(s) && !frontier.contains(s)) {
                    frontier.add(s);
                }
            }

            ++iterations;
        }
    }

    private static void printSearchStatus(long startTime, HashSet<State> explored, Frontier frontier) {
        String statusTemplate = "#Explored: %,8d, #Frontier: %,8d, #Generated: %,8d, Time: %3.3f s\n%s\n";
        double elapsedTime = (System.nanoTime() - startTime) / 1_000_000_000d;
        System.err.format(statusTemplate, explored.size(), frontier.size(), explored.size() + frontier.size(),
                elapsedTime, Memory.stringRep());
    }

    private static void printSearchStatus(long startTime, int explored) {
        String statusTemplate = "#Explored: %,8d, Time: %3.3f s\n%s\n";
        double elapsedTime = (System.nanoTime() - startTime) / 1_000_000_000d;
        System.err.format(statusTemplate, explored, elapsedTime, Memory.stringRep());
    }



    public static void main(String[] args)
            throws IOException {
        // Send client name to server.
        System.out.println("SearchClient");

        // Parse the level.
        BufferedReader serverMessages = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.US_ASCII));
        State initialState = SearchClient.parseLevel(serverMessages);

        // Select search strategy.
        Frontier frontier;
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
                default:
                    frontier = new FrontierBFS();
                    System.err.println("Defaulting to BFS search. Use arguments -bfs, -dfs, -astar, -wastar, or " +
                            "-greedy to set the search strategy.");
            }
        } else {
            frontier = new FrontierBFS();
            System.err.println("Defaulting to BFS search. Use arguments -bfs, -dfs, -astar, -wastar, or -greedy to " +
                    "set the search strategy.");
        }

        // Search for a plan.
        Action[][] plan = null;
        try {
//            plan = SearchClient.search(initialState, frontier);
            MonteCarloTreeSearch monteCarloTreeSearch = new Basic(new UCTSelection(), new AllActionsExpansion(),
                    new AllPairsShortestPath(initialState), new AdditiveBackpropagation());
//            MonteCarloTreeSearch monteCarloTreeSearch = new OneTree(new UCTSelection(), new AllActionsNoDuplicateExpansion(),
//                    new AllPairsShortestPath(initialState), new AdditiveBackpropagation());
            plan = monteCarloTreeSearch.solve(new Node(initialState));
        } catch (OutOfMemoryError ex) {
            System.err.println("Maximum memory usage exceeded.");
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
}
