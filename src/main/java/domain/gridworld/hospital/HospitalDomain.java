package domain.gridworld.hospital;

import client.Client;
import client.Timeout;
import domain.Domain;
import domain.ParseException;
import server.Server;

import java.awt.*;
import java.awt.font.TextLayout;
import java.awt.geom.AffineTransform;
import java.awt.geom.Rectangle2D;
import java.io.*;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.concurrent.TimeUnit;

public final class HospitalDomain
        implements Domain {
    private Path levelFile;
    private StateSequence stateSequence;

    private volatile String clientName = null;

    private long numActions = 0;

    /**
     * Rendering constants.
     */
    private static final Color LETTERBOX_COLOR = Color.BLACK;
    private static final Color GRID_COLOR = Color.DARK_GRAY;
    private static final Color CELL_COLOR = Color.LIGHT_GRAY;
    private static final Color WALL_COLOR = Color.BLACK;
    private static final Color BOX_AGENT_FONT_COLOR = Color.BLACK;
    private static final Color GOAL_COLOR = Colors.UnsolvedGoal;
    private static final Color GOAL_FONT_COLOR = blendColors(GOAL_COLOR, Color.BLACK, 0.7);
    private static final Color GOAL_SOLVED_COLOR = Colors.SolvedGoal;

    private static final double BOX_MARGIN_PERCENT = 0.1;
    private static final double TEXT_MARGIN_PERCENT = 0.2;

    private static final Stroke OUTLINE_STROKE = new BasicStroke(2.0f);
    private static final AffineTransform IDENTITY_TRANSFORM = new AffineTransform();

    @SuppressWarnings("SameParameterValue")
    private static Color blendColors(Color c1, Color c2, double ratio) {
        int r = (int) ((1.0 - ratio) * c1.getRed() + ratio * c2.getRed());
        int g = (int) ((1.0 - ratio) * c1.getGreen() + ratio * c2.getGreen());
        int b = (int) ((1.0 - ratio) * c1.getBlue() + ratio * c2.getBlue());
        return new Color(r, g, b);
    }

    /**
     * Rendering context.
     */
    private int originLeft, originTop;
    private int width, height;
    private int cellSize;
    private int cellBoxMargin;
    // Agent and box letters.
    // FIXME: TextLayout performance sucks. Replace.
    private ArrayList<Agent> agents;

    private TextLayout[] boxLetterText = new TextLayout[26];
    private int[] boxLetterTopOffset = new int[26];
    private int[] boxLetterLeftOffset = new int[26];
    /*
    private TextLayout[] agentLetterText = new TextLayout[10];
    private int[] agentLetterTopOffset = new int[10];
    private int[] agentLetterLeftOffset = new int[10];
    // Agent colors.
    private Color[] agentOutlineColor = new Color[10];
    private Color[] agentArmColor = new Color[10];
     */
    // For drawing arms on agents.
    private Polygon agentArmMove = new Polygon();
    private Polygon agentArmPushPull = new Polygon();
    private AffineTransform agentArmTransform = new AffineTransform();

    // In an interpolation between two states, we track the static and dynamic elements when possible to draw less.
    private boolean staticElementsRendered = false;
    private int numDynamicBoxes;
    private int[] dynamicBoxes = new int[10];
    private int numDynamicAgents;
    private byte[] dynamicAgents = new byte[10];
    private int[] dynamicAgentsBox = new int[10];
    // For each state we cache the box goals which are solved in that state. The state data structures are not designed
    // for that to be efficiently calculable.
    private BitSet[] stateSolvedBoxGoals = new BitSet[1024];

    public HospitalDomain(Path domainFile, boolean isLogFile)
            throws IOException, ParseException {
        this.levelFile = domainFile;
        this.stateSequence = new StateSequence(domainFile, isLogFile);

        if (isLogFile) {
            this.clientName = this.stateSequence.clientName;
            this.numActions = this.stateSequence.getNumStates() - 1;
        }

        //Initate Agents
        agents = new ArrayList<Agent>();
        for (byte i = 0; i < this.stateSequence.numAgents; ++i) {
            agents.add(new Agent(i, this.stateSequence.agentColors[i]));
        }
    }

    @Override
    public void runProtocol(Timeout timeout,
                            long timeoutNS,
                            BufferedInputStream clientIn,
                            BufferedOutputStream clientOut,
                            OutputStream logOut
    ) {
        Client.printDebug("Protocol begun.");

        BufferedReader clientReader
                = new BufferedReader(new InputStreamReader(clientIn, StandardCharsets.US_ASCII.newDecoder()));
        BufferedWriter clientWriter
                = new BufferedWriter(new OutputStreamWriter(clientOut, StandardCharsets.US_ASCII.newEncoder()));
        BufferedWriter logWriter
                = new BufferedWriter(new OutputStreamWriter(logOut, StandardCharsets.US_ASCII.newEncoder()));

        // Read client name. 10 seconds timeout.
        timeout.reset(System.nanoTime(), TimeUnit.SECONDS.toNanos(10));
        String clientMsg;
        try {
            Client.printDebug("Waiting for client name.");
            clientMsg = clientReader.readLine();
        } catch (CharacterCodingException e) {
            Client.printError("Client message not valid ASCII.");
            return;
        } catch (IOException e) {
            // FIXME: What may even cause this? Closing the stream causes readLine to return null rather than throw.
            synchronized (System.out) {
                Client.printError("Unexpected exception while reading client name:");
                e.printStackTrace(System.out);
            }
            return;
        }

        // Check and reset timeout.
        long startNS = System.nanoTime();
        if (!timeout.reset(startNS, timeoutNS)) {
            Client.printError("Timed out while waiting for client name.");
            return;
        }

        // Store client name.
        if (clientMsg != null) {
            Client.printDebug("Received client name: " + clientMsg);
            this.clientName = clientMsg;
        } else {
            Client.printError("Client closed its output stream before sending its name.");
            return;
        }

        // Send level to client and log.
        Client.printDebug("Opening level file: " + this.levelFile);
        try (InputStream levelStream = Files.newInputStream(this.levelFile)) {
            Client.printDebug("Writing level to client and log.");
            try {
                byte[] buffer = new byte[4096];
                int len;
                boolean endNewline = false;
                while ((len = levelStream.readNBytes(buffer, 0, buffer.length)) != 0) {
                    clientOut.write(buffer, 0, len);
                    logOut.write(buffer, 0, len);
                    endNewline = buffer[len - 1] == '\n';
                }
                clientOut.flush();
                logOut.flush();
                if (!endNewline) {
                    clientWriter.newLine();
                    clientWriter.flush();
                    logWriter.newLine();
                    logWriter.flush();
                }
            } catch (IOException e) {
                if (timeout.isExpired()) {
                    Client.printError("Timeout expired while sending level to client.");
                } else {
                    Client.printError("Could not send level to client and/or log.");
                    Client.printError(e.getMessage());
                }
                return;
            }
        } catch (IOException e) {
            Client.printError("Could not open level file.");
            Client.printError(e.getMessage());
            return;
        }

        // Log client name.
        // Has to be logged after level file contents have been logged (first log lines must be domain).
        try {
            Client.printDebug("Logging client name.");
            logWriter.write("#clientname");
            logWriter.newLine();
            logWriter.write(this.clientName);
            logWriter.newLine();
            logWriter.flush();
        } catch (IOException e) {
            Client.printError("Could not write client name to log file.");
            Client.printError(e.getMessage());
            return;
        }

        Client.printDebug("Beginning action/comment message exchanges.");
        long numMessages = 0;
        Action[] jointAction = new Action[this.stateSequence.numAgents];

        try {
            logWriter.write("#actions");
            logWriter.newLine();
            logWriter.flush();
        } catch (IOException e) {
            Client.printError("Could not write to log file.");
            Client.printError(e.getMessage());
            return;
        }

        protocolLoop:
        while (true) {
            if (timeout.isExpired()) {
                Client.printDebug("Client timed out in protocol loop.");
                break;
            }

            // Read client message.
            try {
                clientMsg = clientReader.readLine();
            } catch (CharacterCodingException e) {
                Client.printError("Client message not valid ASCII.");
                return;
            } catch (IOException e) {
                Client.printError("Unexpected exception while reading from client.");
                Client.printError(e.getMessage());
                e.printStackTrace();
                return;
            }
            if (clientMsg == null) {
                if (timeout.isExpired()) {
                    Client.printDebug("Client stream closed after timeout.");
                } else {
                    Client.printDebug("Client closed its output stream.");
                }
                break;
            }

            if (timeout.isExpired()) {
                Client.printDebug("Client timed out in protocol loop.");
                break;
            }

            // Process message.
            ++numMessages;
            if (clientMsg.startsWith("#")) {
                Client.printMessage(clientMsg.substring(1));
            } else {
                // Parse action string.
                String[] actionMsg = clientMsg.split(";");
                if (actionMsg.length != this.stateSequence.numAgents) {
                    Client.printError("Invalid number of agents in joint action:");
                    Client.printError(clientMsg);
                    continue;
                }
                for (int i = 0; i < jointAction.length; ++i) {
                    jointAction[i] = Action.parse(actionMsg[i]);
                    if (jointAction[i] == null) {
                        Client.printError("Invalid joint action:");
                        Client.printError(clientMsg);
                        continue protocolLoop;
                    }
                }

                // Execute action.
                long actionTime = System.nanoTime() - startNS;
                boolean[] result = this.stateSequence.execute(jointAction, actionTime);
                ++this.numActions;

                // Write response.
                try {
                    clientWriter.write(result[0] ? "true" : "false");
                    for (int i = 1; i < result.length; ++i) {
                        clientWriter.write(";");
                        clientWriter.write(result[i] ? "true" : "false");
                    }
                    clientWriter.newLine();
                    clientWriter.flush();
                } catch (IOException e) {
                    // TODO: Happens when client closes before reading responses, then server can't write to the
                    //  client's input stream.
                    Client.printError("Could not write response to client.");
                    Client.printError(e.getMessage());
                    return;
                }

                // Log action.
                try {
                    logWriter.write(Long.toString(actionTime));
                    logWriter.write(":");
                    logWriter.write(clientMsg);
                    logWriter.newLine();
                    logWriter.flush();
                } catch (IOException e) {
                    Client.printError("Could not write to log file.");
                    Client.printError(e.getMessage());
                    return;
                }
            }
        }
        Client.printDebug("Messages exchanged: " + numMessages + ".");

        // Log summary.
        try {
            logWriter.write("#end");
            logWriter.newLine();

            logWriter.write("#solved");
            logWriter.newLine();
            logWriter.write(this.isGoalState(this.getNumStates() - 1) ? "true" : "false");
            logWriter.newLine();

            logWriter.write("#numactions");
            logWriter.newLine();
            logWriter.write(Long.toString(this.numActions));
            logWriter.newLine();

            logWriter.write("#time");
            logWriter.newLine();
            logWriter.write(Long.toString(this.getStateTime(this.getNumStates() - 1)));
            logWriter.newLine();

            logWriter.write("#end");
            logWriter.newLine();
            logWriter.flush();
        } catch (IOException e) {
            Client.printError("Could not write to log file.");
            Client.printError(e.getMessage());
            return;
        }

        Client.printDebug("Protocol finished.");
    }

    @Override
    public void allowDiscardingPastStates() {
        this.stateSequence.allowDiscardingPastStates();
    }

    @Override
    public String getLevelName() {
        return this.stateSequence.getLevelName();
    }

    @Override
    public String getClientName() {
        return this.clientName;
    }

    @Override
    public String[] getStatus() {
        int lastStateID = this.getNumStates() - 1;
        boolean isSolved = this.isGoalState(lastStateID);

        String[] status = new String[3];
        status[0] = String.format("Level solved: %s.", isSolved ? "Yes" : "No");
        status[1] = String.format("Actions used: %d.", this.numActions);
        status[2] = String.format("Last action time: %.3f seconds.", this.getStateTime(lastStateID) / 1_000_000_000d);

        return status;
    }

    private boolean isGoalState(int stateID) {
        boolean isSolved = true;
        State state = this.stateSequence.getState(stateID);
        var solvedBoxGoals = this.getSolvedBoxGoals(stateID);
        if (solvedBoxGoals.nextClearBit(0) != this.stateSequence.numBoxGoals) {
            isSolved = false;
        } else {
            for (byte agent = 0; agent < this.stateSequence.numAgents; ++agent) {
                if (this.stateSequence.agentGoalRows[agent] != -1 &&
                        (this.stateSequence.agentGoalRows[agent] != state.agentRows[agent] ||
                                this.stateSequence.agentGoalCols[agent] != state.agentCols[agent])
                ) {
                    isSolved = false;
                    break;
                }
            }
        }
        return isSolved;
    }

    @Override
    public int getNumStates() {
        return this.stateSequence.getNumStates();
    }

    @Override
    public long getStateTime(int stateID) {
        return this.stateSequence.getStateTime(stateID);
    }

    @Override
    public void renderDomainBackground(Graphics2D g, int width, int height) {
        int numRows = this.stateSequence.numRows;
        int numCols = this.stateSequence.numCols;

        // Determine sizes.
        this.calculateRenderSizes(g, width, height, numRows, numCols);

        // Letterbox.
        g.setColor(LETTERBOX_COLOR);
        g.fillRect(0, 0, width, height);

        // Cell background.
        g.setColor(CELL_COLOR);
        g.fillRect(this.originLeft, this.originTop, this.width, this.height);

        // Grid and walls.
        for (short row = 0; row < numRows; ++row) {
            int top = this.originTop + row * this.cellSize;
            for (short col = 0; col < numCols; ++col) {
                int left = this.originLeft + col * this.cellSize;
                if (this.stateSequence.wallAt(row, col)) {
                    g.setColor(WALL_COLOR);
                    g.fillRect(left, top, this.cellSize, this.cellSize);
                } else {
                    g.setColor(GRID_COLOR);
                    g.drawRect(left, top, this.cellSize - 1, this.cellSize - 1);
                }
            }
        }

        // Goal cells.
        for (int boxGoal = 0; boxGoal < this.stateSequence.numBoxGoals; ++boxGoal) {
            short row = this.stateSequence.boxGoalRows[boxGoal];
            short col = this.stateSequence.boxGoalCols[boxGoal];
            byte boxGoalLetter = this.stateSequence.boxGoalLetters[boxGoal];
            this.drawBoxGoalCell(g, row, col, (char) ('A' + boxGoalLetter), false);
        }
        for (byte agent = 0; agent < this.stateSequence.numAgents; ++agent) {
            short row = this.stateSequence.agentGoalRows[agent];
            short col = this.stateSequence.agentGoalCols[agent];
            if (row != -1) {
                this.drawAgentGoalCell(g, row, col, (char) ('0' + agent), false);
            }
        }
    }

    @Override
    public void renderStateBackground(Graphics2D g, int stateID) {
        State currentState = this.stateSequence.getState(stateID);

        // Highlight solved goal cells.
        BitSet solvedBoxGoals = this.getSolvedBoxGoals(stateID);
        for (int boxGoal = 0; boxGoal < this.stateSequence.numBoxGoals; ++boxGoal) {
            short row = this.stateSequence.boxGoalRows[boxGoal];
            short col = this.stateSequence.boxGoalCols[boxGoal];
            byte boxGoalLetter = this.stateSequence.boxGoalLetters[boxGoal];
            boolean currentSolved = solvedBoxGoals.get(boxGoal);
            if (currentSolved) {
                this.drawBoxGoalCell(g, row, col, (char) ('A' + boxGoalLetter), true);
            }
        }
        for (byte agent = 0; agent < this.stateSequence.numAgents; ++agent) {
            short row = this.stateSequence.agentGoalRows[agent];
            short col = this.stateSequence.agentGoalCols[agent];
            boolean currentSolved = currentState.agentRows[agent] == row && currentState.agentCols[agent] == col;
            if (currentSolved) {
                this.drawAgentGoalCell(g, row, col, (char) ('0' + agent), true);
            }
        }

        // Can we determine static elements? Is the next state known already?
        if (stateID < this.stateSequence.getNumStates() - 1) {
            State nextState = this.stateSequence.getState(stateID + 1);

            this.numDynamicBoxes = 0;
            this.numDynamicAgents = 0;

            // Find dynamic boxes and draw static ones.
            for (int box = 0; box < this.stateSequence.numBoxes; ++box) {
                if (currentState.boxRows[box] == nextState.boxRows[box] &&
                        currentState.boxCols[box] == nextState.boxCols[box]) {
                    byte letter = this.stateSequence.boxLetters[box];
                    int top = this.originTop + currentState.boxRows[box] * this.cellSize;
                    int left = this.originLeft + currentState.boxCols[box] * this.cellSize;
                    this.drawBox(g, top, left, (char) ('A' + letter), this.stateSequence.boxColors[letter]);
                } else {
                    this.dynamicBoxes[this.numDynamicBoxes] = box;
                    ++this.numDynamicBoxes;
                }
            }

            // Find dynamic agents and draw static ones.
            for (byte agent = 0; agent < this.stateSequence.numAgents; ++agent) {
                if (currentState.agentRows[agent] == nextState.agentRows[agent] &&
                        currentState.agentCols[agent] == nextState.agentCols[agent]) {
                    int top = this.originTop + currentState.agentRows[agent] * this.cellSize;
                    int left = this.originLeft + currentState.agentCols[agent] * this.cellSize;
                    this.drawAgent(g, top, left, (char) ('0' + agent), agent);
                } else {
                    this.dynamicAgents[this.numDynamicAgents] = agent;

                    // Determine if the agent is moving a box.
                    this.dynamicAgentsBox[this.numDynamicAgents] = -1; // Assume not.
                    for (int dynamicBox = 0; dynamicBox < this.numDynamicBoxes; ++dynamicBox) {
                        int box = this.dynamicBoxes[dynamicBox];
                        if (nextState.agentRows[agent] == currentState.boxRows[box] &&
                                nextState.agentCols[agent] == currentState.boxCols[box]) {
                            // Pushing this box.
                            this.dynamicAgentsBox[this.numDynamicAgents] = box;
                            break;
                        } else if (currentState.agentRows[agent] == nextState.boxRows[box] &&
                                currentState.agentCols[agent] == nextState.boxCols[box]) {
                            // Pulling this box.
                            this.dynamicAgentsBox[this.numDynamicAgents] = box;
                            break;
                        }
                    }

                    ++this.numDynamicAgents;
                }
            }

            this.staticElementsRendered = true;
        } else {
            // We can't draw anything, because we don't know which elements will have to move in the state transition.
            this.staticElementsRendered = false;
            this.numDynamicBoxes = -1;
            this.numDynamicAgents = -1;
        }
    }

    @Override
    public void renderStateTransition(Graphics2D g, int stateID, double interpolation) {
        if (interpolation < 0.0 || interpolation >= 1.0) {
            Server.printError("Bad interpolation: " + interpolation);
            return;
        }

        State currentState = this.stateSequence.getState(stateID);
        State nextState = interpolation == 0.0 ? currentState : this.stateSequence.getState(stateID + 1);
        BitSet currentSolvedBoxGoals = this.getSolvedBoxGoals(stateID);
        BitSet nextSolvedBoxGoals = interpolation == 0.0 ? currentSolvedBoxGoals : this.getSolvedBoxGoals(stateID + 1);

        // Un-highlight goal cells that were solved, but are unsolved in this transition.
        for (int boxGoal = 0; boxGoal < this.stateSequence.numBoxGoals; ++boxGoal) {
            short row = this.stateSequence.boxGoalRows[boxGoal];
            short col = this.stateSequence.boxGoalCols[boxGoal];
            byte boxGoalLetter = this.stateSequence.boxGoalLetters[boxGoal];
            boolean currentSolved = currentSolvedBoxGoals.get(boxGoal);
            boolean nextSolved = nextSolvedBoxGoals.get(boxGoal);
            if (currentSolved && !nextSolved) {
                this.drawBoxGoalCell(g, row, col, (char) ('A' + boxGoalLetter), false);
            }
        }
        for (byte agent = 0; agent < this.stateSequence.numAgents; ++agent) {
            short row = this.stateSequence.agentGoalRows[agent];
            short col = this.stateSequence.agentGoalCols[agent];
            boolean currentSolved = currentState.agentRows[agent] == row && currentState.agentCols[agent] == col;
            boolean nextSolved = nextState.agentRows[agent] == row && nextState.agentCols[agent] == col;
            if (currentSolved && !nextSolved) {
                this.drawAgentGoalCell(g, row, col, (char) ('0' + agent), false);
            }
        }

        // Have we determined the dynamic entities in the transition? Otherwise do it now.
        if (interpolation != 0.0 && this.numDynamicAgents == -1) {
            this.numDynamicBoxes = 0;
            this.numDynamicAgents = 0;

            // Find dynamic boxes.
            for (int box = 0; box < this.stateSequence.numBoxes; ++box) {
                if (currentState.boxRows[box] != nextState.boxRows[box] ||
                        currentState.boxCols[box] != nextState.boxCols[box]) {
                    this.dynamicBoxes[this.numDynamicBoxes] = box;
                    ++this.numDynamicBoxes;
                }
            }

            // Find dynamic agents.
            for (byte agent = 0; agent < this.stateSequence.numAgents; ++agent) {
                if (currentState.agentRows[agent] != nextState.agentRows[agent] ||
                        currentState.agentCols[agent] != nextState.agentCols[agent]) {
                    this.dynamicAgents[this.numDynamicAgents] = agent;

                    // Determine if the agent is moving a box.
                    this.dynamicAgentsBox[this.numDynamicAgents] = -1; // Assume not.
                    for (int dynamicBox = 0; dynamicBox < this.numDynamicBoxes; ++dynamicBox) {
                        int box = this.dynamicBoxes[dynamicBox];
                        if (nextState.agentRows[agent] == currentState.boxRows[box] &&
                                nextState.agentCols[agent] == currentState.boxCols[box]) {
                            // Pushing this box.
                            this.dynamicAgentsBox[this.numDynamicAgents] = box;
                            break;
                        } else if (currentState.agentRows[agent] == nextState.boxRows[box] &&
                                currentState.agentCols[agent] == nextState.boxCols[box]) {
                            // Pulling this box.
                            this.dynamicAgentsBox[this.numDynamicAgents] = box;
                            break;
                        }
                    }

                    ++this.numDynamicAgents;
                }
            }
        }

        // Draw arms on agents (under agent and box, so we can rely on overlap to form desired shapes).
        if (interpolation != 0.0) {
            for (byte dynamicAgent = 0; dynamicAgent < this.numDynamicAgents; ++dynamicAgent) {
                byte agent = this.dynamicAgents[dynamicAgent];
                int box = this.dynamicAgentsBox[dynamicAgent];
                if (box != -1) {
                    // Push/Pull.
                    // Agent position.
                    int cTop = this.originTop + currentState.agentRows[agent] * this.cellSize;
                    int cLeft = this.originLeft + currentState.agentCols[agent] * this.cellSize;
                    int nTop = this.originTop + nextState.agentRows[agent] * this.cellSize;
                    int nLeft = this.originLeft + nextState.agentCols[agent] * this.cellSize;
                    int iTop = (int) (cTop + (nTop - cTop) * interpolation);
                    int iLeft = (int) (cLeft + (nLeft - cLeft) * interpolation);
                    // Box position.
                    int bcTop = this.originTop + currentState.boxRows[box] * this.cellSize;
                    int bcLeft = this.originLeft + currentState.boxCols[box] * this.cellSize;
                    int bnTop = this.originTop + nextState.boxRows[box] * this.cellSize;
                    int bnLeft = this.originLeft + nextState.boxCols[box] * this.cellSize;
                    int biTop = (int) (bcTop + (bnTop - bcTop) * interpolation);
                    int biLeft = (int) (bcLeft + (bnLeft - bcLeft) * interpolation);

                    double direction = Math.atan2(biTop - iTop, biLeft - iLeft);
                    this.drawAgentArm(g, this.agentArmPushPull, iTop, iLeft, direction, agent);
                } else {
                    // Move.
                    int cTop = this.originTop + currentState.agentRows[agent] * this.cellSize;
                    int cLeft = this.originLeft + currentState.agentCols[agent] * this.cellSize;
                    int nTop = this.originTop + nextState.agentRows[agent] * this.cellSize;
                    int nLeft = this.originLeft + nextState.agentCols[agent] * this.cellSize;
                    int iTop = (int) (cTop + (nTop - cTop) * interpolation);
                    int iLeft = (int) (cLeft + (nLeft - cLeft) * interpolation);
                    double direction = Math.atan2(nTop - cTop, nLeft - cLeft);
                    this.drawAgentArm(g, this.agentArmMove, iTop, iLeft, direction, agent);
                }
            }
        }

        if (this.staticElementsRendered) {
            // Draw dynamic boxes.
            for (int dynamicBox = 0; dynamicBox < this.numDynamicBoxes; ++dynamicBox) {
                int box = this.dynamicBoxes[dynamicBox];
                byte letter = this.stateSequence.boxLetters[box];
                int cTop = this.originTop + currentState.boxRows[box] * this.cellSize;
                int cLeft = this.originLeft + currentState.boxCols[box] * this.cellSize;
                int nTop = this.originTop + nextState.boxRows[box] * this.cellSize;
                int nLeft = this.originLeft + nextState.boxCols[box] * this.cellSize;
                int iTop = (int) (cTop + (nTop - cTop) * interpolation);
                int iLeft = (int) (cLeft + (nLeft - cLeft) * interpolation);
                this.drawBox(g, iTop, iLeft, (char) ('A' + letter), this.stateSequence.boxColors[letter]);
            }

            // Draw dynamic agents.
            for (byte dynamicAgent = 0; dynamicAgent < this.numDynamicAgents; ++dynamicAgent) {
                byte agent = this.dynamicAgents[dynamicAgent];
                int cTop = this.originTop + currentState.agentRows[agent] * this.cellSize;
                int cLeft = this.originLeft + currentState.agentCols[agent] * this.cellSize;
                int nTop = this.originTop + nextState.agentRows[agent] * this.cellSize;
                int nLeft = this.originLeft + nextState.agentCols[agent] * this.cellSize;
                int iTop = (int) (cTop + (nTop - cTop) * interpolation);
                int iLeft = (int) (cLeft + (nLeft - cLeft) * interpolation);
                this.drawAgent(g, iTop, iLeft, (char) ('0' + agent), agent);
            }
        } else {
            // Draw all boxes.
            for (int box = 0; box < this.stateSequence.numBoxes; ++box) {
                byte letter = this.stateSequence.boxLetters[box];
                int cTop = this.originTop + currentState.boxRows[box] * this.cellSize;
                int cLeft = this.originLeft + currentState.boxCols[box] * this.cellSize;
                int nTop = this.originTop + nextState.boxRows[box] * this.cellSize;
                int nLeft = this.originLeft + nextState.boxCols[box] * this.cellSize;
                int iTop = (int) (cTop + (nTop - cTop) * interpolation);
                int iLeft = (int) (cLeft + (nLeft - cLeft) * interpolation);
                this.drawBox(g, iTop, iLeft, (char) ('A' + letter), this.stateSequence.boxColors[letter]);
            }

            // Draw all agents.
            for (byte agent = 0; agent < this.stateSequence.numAgents; ++agent) {
                int cTop = this.originTop + currentState.agentRows[agent] * this.cellSize;
                int cLeft = this.originLeft + currentState.agentCols[agent] * this.cellSize;
                int nTop = this.originTop + nextState.agentRows[agent] * this.cellSize;
                int nLeft = this.originLeft + nextState.agentCols[agent] * this.cellSize;
                int iTop = (int) (cTop + (nTop - cTop) * interpolation);
                int iLeft = (int) (cLeft + (nLeft - cLeft) * interpolation);
                this.drawAgent(g, iTop, iLeft, (char) ('0' + agent), agent);
            }
        }
    }

    private void calculateRenderSizes(Graphics2D g, int width, int height, int numRows, int numCols) {
        this.cellSize = Math.min(width / numCols, height / numRows);

        int excessWidth = width - numCols * this.cellSize;
        int excessHeight = height - numRows * this.cellSize;
        this.originLeft = excessWidth / 2;
        this.originTop = excessHeight / 2;
        this.width = width - excessWidth;
        this.height = height - excessHeight;

        this.cellBoxMargin = (int) (this.cellSize * BOX_MARGIN_PERCENT);
        int cellTextMargin = (int) (this.cellSize * TEXT_MARGIN_PERCENT);

        // Determine font size.
        var fontRenderContext = g.getFontRenderContext();
        int fontSize = 0;
        Font curFont;
        Font nextFont = new Font(null, Font.BOLD, fontSize);
        Rectangle bounds;
        Rectangle2D bounds2;
        do {
            curFont = nextFont;
            ++fontSize;
            nextFont = new Font(null, Font.BOLD, fontSize);
            bounds2 = nextFont.getStringBounds("W", fontRenderContext);
            // FIXME: Holy shit, creating a TextLayout object is SLOW!
            long t1 = System.nanoTime();
            var text = new TextLayout("W", nextFont, fontRenderContext); // Using W because it's wide.
            Server.printDebug(String.format("fontSize: %d us.", (System.nanoTime() - t1) / 1000));
            bounds = text.getPixelBounds(fontRenderContext, 0, 0);
        } while (bounds.width < this.cellSize - 2 * cellTextMargin &&
                bounds.height < this.cellSize - 2 * cellTextMargin);

//        System.err.println(this.cellSize - 2 * cellTextMargin);
//        System.err.println(bounds);
//        System.err.println(bounds2);
//        System.err.println(g.getDeviceConfiguration());

        long t1 = System.nanoTime();
        // Layout box and agent letters.
        for (int letter = 0; letter < 26; ++letter) {
            // FIXME: Holy shit, creating a TextLayout object is SLOW!
            this.boxLetterText[letter] = new TextLayout(Character.toString('A' + letter), curFont, fontRenderContext);
            Rectangle bound = this.boxLetterText[letter].getPixelBounds(fontRenderContext, 0, 0);
            int size = this.cellSize - 2 * cellTextMargin;
            this.boxLetterTopOffset[letter] = cellTextMargin + size - (size - bound.height) / 2;
            this.boxLetterLeftOffset[letter] = cellTextMargin + (size - bound.width) / 2 - bound.x;
        }

        for(Agent agent : agents) {
            // FIXME: Holy shit, creating a TextLayout object is SLOW!
            agent.setLetterText(new TextLayout(Character.toString('0' + agent.id), curFont, fontRenderContext));
            Rectangle bound = agent.getLetterText().getPixelBounds(fontRenderContext, 0, 0);
            int size = this.cellSize - 2 * cellTextMargin;
            agent.setLetterTopOffset(cellTextMargin + size - (size - bound.height) / 2);
            agent.setLetterLeftOffset(cellTextMargin + (size - bound.width) / 2 - bound.x);
        }
        Server.printDebug(String.format("layoutLetters: %d ms.", (System.nanoTime() - t1) / 1000000));


        // Agent move arm shape.
        // A triangle "pointing" left and with one point on (0,0) and two other points on either side of the x-axis.
        int armLength = this.cellSize / 2 - 1;
        int armHeight = (int) (this.cellSize * 0.60);
        this.agentArmMove.reset();
        this.agentArmMove.addPoint(0, 0);
        this.agentArmMove.addPoint(armLength, -armHeight / 2);
        this.agentArmMove.addPoint(armLength, armHeight / 2);
        this.agentArmMove.addPoint(0, 0);

        // Agent push/pull arm shape.
        // A triangle "pointing" left and with one point on (0,0) and two other points on either side of the x-axis.
        armLength = (int) (this.cellSize / Math.sqrt(2.0));
        armHeight = (int) ((this.cellSize - 2 * this.cellBoxMargin) / Math.sqrt(2.0));
        this.agentArmPushPull.reset();
        this.agentArmPushPull.addPoint(0, 0);
        this.agentArmPushPull.addPoint(armLength, -armHeight / 2);
        this.agentArmPushPull.addPoint(armLength, armHeight / 2);
        this.agentArmPushPull.addPoint(0, 0);
    }

    private BitSet getSolvedBoxGoals(int stateID) {
        if (stateID >= this.stateSolvedBoxGoals.length) {
            int newSize = Math.max(this.stateSolvedBoxGoals.length * 2, stateID + 1);
            this.stateSolvedBoxGoals = Arrays.copyOf(this.stateSolvedBoxGoals, newSize);
        }
        BitSet solvedBoxGoals = this.stateSolvedBoxGoals[stateID];
        if (solvedBoxGoals == null) {
            State state = this.stateSequence.getState(stateID);
            solvedBoxGoals = new BitSet(this.stateSequence.numBoxGoals);

            // TODO: Is this too inefficient?
            //       With N boxes and M box goal cells, where 0 <= M <= N, the complexity is O(N*log(M)).
            //       If boxes were sorted, we could have O(M*log(N)), but the sorting itself
            //       eclipses that (unless maintained in each state).
            for (int box = 0; box < this.stateSequence.numBoxes; ++box) {
                short boxRow = state.boxRows[box];
                short boxCol = state.boxCols[box];
                byte boxLetter = this.stateSequence.boxLetters[box];
                int boxGoal = this.stateSequence.findBoxGoal(boxRow, boxCol);
                if (boxGoal != -1 && boxLetter == this.stateSequence.boxGoalLetters[boxGoal]) {
                    solvedBoxGoals.set(boxGoal);
                }
            }

            this.stateSolvedBoxGoals[stateID] = solvedBoxGoals;
        }
        return solvedBoxGoals;
    }

    private void drawBoxGoalCell(Graphics2D g, short row, short col, char letter, boolean solved) {
        int top = this.originTop + row * this.cellSize;
        int left = this.originLeft + col * this.cellSize;
        int size = this.cellSize - 2;
        g.setColor(solved ? GOAL_SOLVED_COLOR : GOAL_COLOR);
        g.fillRect(left + 1, top + 1, size, size);

        // No need to draw text if cell is solved, since box will be drawn on top of text anyway.
        if (!solved) {
            TextLayout letterText = this.boxLetterText[letter - 'A'];
            int letterTopOffet = this.boxLetterTopOffset[letter - 'A'];
            int letterLeftOffet = this.boxLetterLeftOffset[letter - 'A'];
            g.setColor(GOAL_FONT_COLOR);
            letterText.draw(g, left + letterLeftOffet, top + letterTopOffet);
        }
    }

    private void drawAgentGoalCell(Graphics2D g, short row, short col, char letter, boolean solved) {
        int top = this.originTop + row * this.cellSize;
        int left = this.originLeft + col * this.cellSize;
        int size = this.cellSize - 2;
        g.setColor(solved ? GOAL_SOLVED_COLOR : GOAL_COLOR);
        g.fillOval(left + 1, top + 1, size, size);

        // No need to draw text if cell is solved, since agent will be drawn on top of text anyway.
        if (!solved) {
            //TODO: Make sure this works - Fredrik
            var agent = agents.get(letter - '0');
            TextLayout letterText = agent.getLetterText();
            int letterTopOffset = agent.getLetterTopOffset();
            int letterLeftOffset = agent.getLetterLeftOffset();
            g.setColor(GOAL_FONT_COLOR);
            letterText.draw(g, left + letterLeftOffset, top + letterTopOffset);
            g.drawString("", 0, 0);
        }
    }

    private void drawBox(Graphics2D g, int top, int left, char letter, Color color) {
        int size = this.cellSize - 2 * this.cellBoxMargin;
        g.setColor(color);
        g.fillRect(left + this.cellBoxMargin, top + this.cellBoxMargin, size, size);

        TextLayout letterText = this.boxLetterText[letter - 'A'];
        int letterTopOffet = this.boxLetterTopOffset[letter - 'A'];
        int letterLeftOffet = this.boxLetterLeftOffset[letter - 'A'];
        g.setColor(BOX_AGENT_FONT_COLOR);
        letterText.draw(g, left + letterLeftOffet, top + letterTopOffet);
    }

    private void drawAgent(Graphics2D g, int top, int left, char letter, byte agentid) {
        int size = this.cellSize - 2 * this.cellBoxMargin;

        // Agent fill.
        g.setColor(this.stateSequence.agentColors[agentid]);
        g.fillOval(left + this.cellBoxMargin, top + this.cellBoxMargin, size, size);

        // Agent outline.
//        g.setColor(this.agentOutlineColor[agent]);
//        Stroke stroke = g.getStroke();
//        g.setStroke(OUTLINE_STROKE);
//        g.drawOval(left + BOX_MARGIN, top + BOX_MARGIN, size, size);
//        g.setStroke(stroke);

        // Agent letter.
        Agent agent = agents.get(letter - '0');
        //
        TextLayout letterText = agent.getLetterText();
        int letterTopOffet = agent.getLetterTopOffset();
        int letterLeftOffet = agent.getLetterLeftOffset();
        g.setColor(BOX_AGENT_FONT_COLOR);
        letterText.draw(g, left + letterLeftOffet, top + letterTopOffet);
        g.drawString("W", 0, 0);
    }

    private void drawAgentArm(Graphics2D g, Polygon armShape, int top, int left, double rotation, byte agentid) {
        int armTop = top + this.cellSize / 2;
        int armLeft = left + this.cellSize / 2;
        this.setArmTransform(armTop, armLeft, rotation);
        g.setTransform(this.agentArmTransform);

        //Get Agent
        var agent = agents.get(agentid);
        // Arm fill.
        g.setColor(agent.getArmColor());
        g.fillPolygon(armShape);

        // Arm outline.
        g.setColor(agent.getOutlineColor());
        Stroke stroke = g.getStroke();
        g.setStroke(OUTLINE_STROKE);
        g.drawPolygon(armShape);
        g.setStroke(stroke);

        g.setTransform(IDENTITY_TRANSFORM);
    }

    private void setArmTransform(int top, int left, double rotation) {
        double cos = Math.cos(rotation);
        double sin = Math.sin(rotation);
        this.agentArmTransform.setTransform(cos, sin, -sin, cos, left, top);
    }
}
