package searchclient;

import shared.Action;
import shared.Farge;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Locale;

public class SearchClient {
    public static State parseLevel(BufferedReader serverMessages) throws IOException {
        // We can assume that the level file is conforming to specification, since the server verifies this.
        // Read domain.
        System.err.println("Started reading level");

        serverMessages.readLine(); // #domain
        serverMessages.readLine(); // hospital

        // Read Level name.
        serverMessages.readLine(); // #levelname
        serverMessages.readLine(); // <name>

        // Read colors.
        serverMessages.readLine(); // #colors
        Farge[] agentColors = new Farge[10];
        Farge[] boxColors = new Farge[26];
        String line = serverMessages.readLine();
        System.err.println("line:");
        System.err.println(line);
        while (!line.startsWith("#")) {
            String[] split = line.split(":");
            Farge colors = Farge.fromString(split[0].strip());
            String[] entities = split[1].split(",");
            for (String entity : entities) {
                char c = entity.strip().charAt(0);
                if ('0' <= c && c <= '9') {
                    agentColors[c - '0'] = colors;
                } else if ('A' <= c && c <= 'Z') {
                    boxColors[c - 'A'] = colors;
                }
            }
            line = serverMessages.readLine();
            System.err.println(line);
        }

        // Read initial state.
        // line is currently "#initial".
        int numAgents = 0;
        int[] agentRows = new int[10];
        int[] agentCols = new int[10];
        boolean[][] walls = new boolean[75][75];
        char[][] boxes = new char[75][75];
        line = serverMessages.readLine();
        System.err.println(line);
        int row = 0;
        while (!line.startsWith("#")) {
            for (int col = 0; col < line.length(); ++col) {
                char c = line.charAt(col);

                if ('0' <= c && c <= '9') {
                    agentRows[c - '0'] = row;
                    agentCols[c - '0'] = col;
                    ++numAgents;
                } else if ('A' <= c && c <= 'Z') {
                    boxes[row][col] = c;
                } else if (c == '+') {
                    walls[row][col] = true;
                }
            }

            ++row;
            line = serverMessages.readLine();
            System.err.println(line);
        }
        agentRows = Arrays.copyOf(agentRows, numAgents);
        agentCols = Arrays.copyOf(agentCols, numAgents);

        // Read goal state.
        // line is currently "#goal".
        char[][] goals = new char[75][75];
        line = serverMessages.readLine();
        System.err.println(line);
        row = 0;
        while (!line.startsWith("#")) {
            for (int col = 0; col < line.length(); ++col) {
                char c = line.charAt(col);

                if (('0' <= c && c <= '9') || ('A' <= c && c <= 'Z')) {
                    goals[row][col] = c;
                }
            }

            ++row;
            line = serverMessages.readLine();
            System.err.println(line);
        }

        // End.
        // line is currently "#end".
        System.err.println("Finished reading level");
        return new State(agentRows, agentCols, agentColors, walls, boxes, boxColors, goals);
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
            frontier = new FrontierBestFirst(new HeuristicGreedy(initialState));
            System.err.println("Defaulting to BFS search. Use arguments -bfs, -dfs, -astar, -wastar, or -greedy to " +
                    "set the search strategy.");
        }

        // Search for a plan.
        Action[][] plan;
        try {
            plan = SearchClient.search(initialState, frontier);
        } catch (OutOfMemoryError ex) {
            System.err.println("Maximum memory usage exceeded.");
            plan = null;
        }

        // Print plan to server.
        if (plan == null) {
            System.err.println("Unable to solve level.");
            System.exit(0);
        } else {
            System.err.format("Found solution of length %d.\n", plan.length);

            for (Action[] jointAction : plan) {
                System.out.print(jointAction[0].name);
                for (int action = 1; action < jointAction.length; ++action) {
                    System.out.print(";");
                    System.out.print(jointAction[action].name);
                }
                System.out.println();
                // We must read the server's response to not fill up the stdin buffer and block the server.
                serverMessages.readLine();
            }
        }
    }
}
