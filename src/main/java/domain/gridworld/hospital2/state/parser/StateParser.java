package domain.gridworld.hospital2.state.parser;

import domain.ParseException;
import domain.gridworld.hospital2.Colors;
import domain.gridworld.hospital2.Object;
import domain.gridworld.hospital2.state.State;
import domain.gridworld.hospital2.state.StaticState;
import lombok.Getter;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.awt.*;
import java.io.IOException;
import java.io.LineNumberReader;
import java.nio.charset.MalformedInputException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

public class StateParser {
    private Logger serverLogger = LogManager.getLogger("server");

    private Path domainFile;
    private boolean isReplay;
    private Color[] agentColors;
    private Color[] boxColors;

    @Getter private StaticState staticState;
    @Getter private State state;

    public StateParser(Path domainFile, boolean isReplay) {
        this.domainFile = domainFile;
        this.isReplay = isReplay;

        this.agentColors = new Color[10];
        this.boxColors = new Color[10];

        this.staticState = new StaticState();
        this.state = new State(new ArrayList<>(), new ArrayList<>(), 0);
    }

    public void parse() throws IOException, ParseException {
        var tStart = System.nanoTime();
        try (LineNumberReader levelReader = new LineNumberReader(Files.newBufferedReader(domainFile,
                StandardCharsets.US_ASCII))) {
            try {
                // Skip the domain type lines.
                levelReader.readLine();
                levelReader.readLine();

                String line = levelReader.readLine();
                if (line == null || !line.equals("#levelname")) {
                    throw new ParseException("Expected beginning of level name section (#levelname).",
                            levelReader.getLineNumber());
                }
                line = this.parseNameSection(levelReader);

                if (line == null || !line.equals("#colors")) {
                    throw new ParseException("Expected beginning of color section (#colors).",
                            levelReader.getLineNumber());
                }
                line = this.parseColorsSection(levelReader);

                if (!line.equals("#initial")) {
                    throw new ParseException("Expected beginning of initial state section (#initial).",
                            levelReader.getLineNumber());
                }
                line = this.parseInitialSection(levelReader);

                if (!line.equals("#goal")) {
                    throw new ParseException("Expected beginning of goal state section (#goal).",
                            levelReader.getLineNumber());
                }
                line = this.parseGoalSection(levelReader);

//                // Initial and goal states loaded; check that states are legal.
//                checkObjectsEnclosedInWalls();

                if (!line.stripTrailing().equalsIgnoreCase("#end")) {
                    throw new ParseException("Expected end section (#end).", levelReader.getLineNumber());
                }
                line = this.parseEndSection(levelReader);
//
//                // If this is a log file, then parse additional sections.
//                if (isReplay) {
//                    // Parse client name.
//                    if (!line.stripTrailing().equalsIgnoreCase("#clientname")) {
//                        throw new ParseException("Expected client name section (#clientname).",
//                                levelReader.getLineNumber());
//                    }
//                    line = this.parseClientNameSection(levelReader);
//
//                    // Parse and simulate actions.
//                    if (!line.stripTrailing().equalsIgnoreCase("#actions")) {
//                        throw new ParseException("Expected actions section (#actions).", levelReader.getLineNumber());
//                    }
//                    line = this.parseActionsSection(levelReader);
//
//                    if (!line.stripTrailing().equalsIgnoreCase("#end")) {
//                        throw new ParseException("Expected end section (#end).", levelReader.getLineNumber());
//                    }
//                    line = this.parseEndSection(levelReader);
//
//                    // Parse summary to check if it is consistent with simulation.
//                    if (!line.stripTrailing().equalsIgnoreCase("#solved")) {
//                        throw new ParseException("Expected solved section (#solved).", levelReader.getLineNumber());
//                    }
//                    line = this.parseSolvedSection(levelReader);
//
//                    if (!line.stripTrailing().equalsIgnoreCase("#numactions")) {
//                        throw new ParseException("Expected numactions section (#numactions).", levelReader.getLineNumber());
//                    }
//                    line = this.parseNumActionsSection(levelReader);
//
//                    if (!line.stripTrailing().equalsIgnoreCase("#time")) {
//                        throw new ParseException("Expected time section (#time).", levelReader.getLineNumber());
//                    }
//                    line = this.parseTimeSection(levelReader);
//
//                    if (!line.stripTrailing().equalsIgnoreCase("#end")) {
//                        throw new ParseException("Expected end section (#end).", levelReader.getLineNumber());
//                    }
//                    line = this.parseEndSection(levelReader);
//                }

                if (line != null) {
                    throw new ParseException("Expected no more content after end section.",
                            levelReader.getLineNumber());
                }
            } catch (MalformedInputException e) {
                throw new ParseException("Level file content not valid ASCII.", levelReader.getLineNumber());
            }
        }
        var tEnd = System.nanoTime();
        serverLogger.debug(String.format("Parsing time: %.3f ms.", (tEnd - tStart) / 1_000_000_000.0));
    }

    private String parseNameSection(LineNumberReader levelReader)
            throws IOException, ParseException {
        String line = levelReader.readLine();
        if (line == null) {
            throw new ParseException("Expected a level name, but reached end of file.", levelReader.getLineNumber());
        }
        if (line.isBlank()) {
            throw new ParseException("Level name can not be blank.", levelReader.getLineNumber());
        }
        this.staticState.setLevelName(line);
        line = levelReader.readLine();
        return line;
    }

    private String parseColorsSection(LineNumberReader levelReader)
            throws IOException, ParseException {
        while (true) {
            String line = levelReader.readLine();
            if (line == null) {
                throw new ParseException("Expected more color lines or end of color section, but reached end of file.",
                        levelReader.getLineNumber());
            }

            if (line.length() > 0 && line.charAt(0) == '#') {
                return line;
            }

            String[] split = line.split(":");
            if (split.length < 1) {
                throw new ParseException("Invalid color line syntax - missing a colon?", levelReader.getLineNumber());
            }
            if (split.length > 2) {
                throw new ParseException("Invalid color line syntax - too many colons?", levelReader.getLineNumber());
            }

            String colorName = split[0].strip().toLowerCase(java.util.Locale.ROOT);
            Color color = Colors.fromString(colorName);
            if (color == null) {
                throw new ParseException(String.format("Invalid color name: '%s'.", colorName),
                        levelReader.getLineNumber());
            }

            String[] symbols = split[1].split(",");
            for (String symbol : symbols) {
                symbol = symbol.strip();
                if (symbol.isEmpty()) {
                    throw new ParseException("Missing agent or box specifier between commas.",
                            levelReader.getLineNumber());
                }
                if (symbol.length() > 1) {
                    throw new ParseException(String.format("Invalid agent or box symbol: '%s'.", symbol),
                            levelReader.getLineNumber());
                }
                char s = symbol.charAt(0);
                if ('0' <= s && s <= '9') {
                    if (this.agentColors[s - '0'] != null) {
                        throw new ParseException(String.format("Agent '%s' already has a color specified.", s),
                                levelReader.getLineNumber());
                    }
                    this.agentColors[s - '0'] = color;
                } else if ('A' <= s && s <= 'Z') {
                    if (this.boxColors[s - 'A'] != null) {
                        throw new ParseException(String.format("Box '%s' already has a color specified.", s),
                                levelReader.getLineNumber());
                    }
                    this.boxColors[s - 'A'] = color;
                } else {
                    throw new ParseException(String.format("Invalid agent or box symbol: '%s'.", s),
                            levelReader.getLineNumber());
                }
            }
        }
    }

    private String parseInitialSection(LineNumberReader levelReader)
            throws IOException, ParseException {
        short numRows = 0;
        List<List<Boolean>> map = new ArrayList<>();
        List<Object> boxes = new ArrayList<>();
        List<Object> agents = new ArrayList<>(10);

        // Parse level and accumulate walls, agents, and boxes.
        String line;
        while (true) {
            line = levelReader.readLine();
            if (line == null) {
                throw new ParseException(
                        "Expected more initial state lines or end of initial state section, but reached end of file.",
                        levelReader.getLineNumber());
            }

            if (line.length() > 0 && line.charAt(0) == '#') {
                break;
            }

            line = this.stripTrailingSpaces(line);

            if (!line.startsWith("+")) {
                throw new ParseException(
                        "Line did not start with a wall.",
                        levelReader.getLineNumber());
            }

            if (line.length() > Short.MAX_VALUE) {
                throw new ParseException(
                        String.format("Initial state too large. Width greater than %s.", Short.MAX_VALUE),
                        levelReader.getLineNumber());
            }

            if (line.length() >= this.staticState.getNumCols()) {
                this.staticState.setNumCols((short) line.length());
            }
            List<Boolean> mapRow = new ArrayList<>(line.length());
            for (short col = 0; col < line.length(); ++col) {
                char c = line.charAt(col);
                if ('0' <= c && c <= '9') {
                    // Agent.
                    int id = c - '0';
                    if (agents.stream().anyMatch(a -> a.getId() == id)) {
                        throw new ParseException(
                                String.format("Agent '%s' appears multiple times in initial state.", c),
                                levelReader.getLineNumber());
                    }
                    Color agentColor = this.agentColors[id];
                    if (agentColor == null) {
                        throw new ParseException(String.format("Agent '%s' has no color specified.", c),
                                levelReader.getLineNumber());
                    }
                    agents.add(new Object(id, c, numRows, col, agentColor));
                } else if ('A' <= c && c <= 'Z') {
                    // Box.
                    int id = c - 'A';
                    Color boxColor = this.boxColors[id];
                    if (boxColor == null) {
                        throw new ParseException(String.format("Box '%s' has no color specified.", c),
                                levelReader.getLineNumber());
                    }
                    boxes.add(new Object(id, c, numRows, col, boxColor));
                } else if (c != '+' && c != ' ') {
                    throw new ParseException(String.format("Invalid character '%s' in column %s.", c, col),
                            levelReader.getLineNumber());
                }
                mapRow.add(c == ' ' || '0' <= c && c <= '9' || 'A' <= c);
            }
            map.add(mapRow);
            numRows++;
            if (numRows < 0) {
                throw new ParseException(
                        String.format("Initial state too large. Height greater than %s.", Short.MAX_VALUE),
                        levelReader.getLineNumber());
            }
        }
        this.staticState.setNumRows(numRows);
        this.staticState.setMap(map);

        this.state.setBoxes(boxes.stream().sorted(Comparator.comparingInt(Object::getId)).collect(Collectors.toList()));
        this.state.setAgents(agents.stream().sorted(Comparator.comparingInt(Object::getId)).collect(Collectors.toList()));

        if (agents.isEmpty()) {
            throw new ParseException("Level contains no agents.", levelReader.getLineNumber());
        }

        return line;
    }

    private String parseGoalSection(LineNumberReader levelReader)
            throws IOException, ParseException {
        short row = 0;

        List<Object> agentGoals = new ArrayList<>(this.state.getAgents().size());
        List<Object> boxGoals = new ArrayList<>(this.state.getBoxes().size());

        String line;
        while (true) {
            line = levelReader.readLine();
            if (line == null) {
                throw new ParseException(
                        "Expected more goal state lines or end of goal state section, but reached end of file.",
                        levelReader.getLineNumber());
            }

            if (line.length() > 0 && line.charAt(0) == '#') {
                if (row != this.staticState.getNumRows()) {
                    throw new ParseException(
                            "Goal state must have the same number of rows as the initial state, but has too few.",
                            levelReader.getLineNumber());
                }
                break;
            }

            if (row == this.staticState.getNumRows()) {
                throw new ParseException(
                        "Goal state must have the same number of rows as the initial state, but has too many.",
                        levelReader.getLineNumber());
            }

            line = this.stripTrailingSpaces(line);

            if (!line.startsWith("+")) {
                throw new ParseException(
                        "Line did not start with a wall.",
                        levelReader.getLineNumber());
            }

            if (line.length() > Short.MAX_VALUE) {
                throw new ParseException(String.format("Goal state too large. Width greater than %s.", Short.MAX_VALUE),
                        levelReader.getLineNumber());
            }

            if (line.length() > this.staticState.getNumCols()) {
                throw new ParseException("Goal state can not have more columns than the initial state.",
                        levelReader.getLineNumber());
            }

            short col = 0;
            for (; col < line.length(); ++col) {
                char c = line.charAt(col);
                if (c == '+') {
                    // Wall.
                    if (!this.staticState.isWall(row, col)) {
                        // Which doesn't match a wall in the initial state.
                        throw new ParseException(
                                String.format("Initial state has no wall at column %d, but goal state does.", col),
                                levelReader.getLineNumber());
                    }
                } else if (this.staticState.isWall(row, col)) {
                    // Missing wall compared to the initial state.
                    throw new ParseException(
                            String.format("Goal state not matching initial state's wall on column %d.", col),
                            levelReader.getLineNumber());
                } else if ('0' <= c && c <= '9') {
                    // Agent.
                    int id = c - '0';
                    if (id >= this.state.getAgents().size()) {
                        throw new ParseException(
                                String.format("Goal state has agent '%s' who does not appear in the initial state.", c),
                                levelReader.getLineNumber());
                    }
                    if (agentGoals.stream().anyMatch(g -> g.getId() == id)) {
                        throw new ParseException(String.format("Agent '%s' appears multiple times in goal state.", c),
                                levelReader.getLineNumber());
                    }
                    agentGoals.add(new Object(id, c, row, col, null));
                } else if ('A' <= c && c <= 'Z') {
                    // Box.
                    int id = c - 'A';
                    boxGoals.add(new Object(id, c, row, col, null));
                } else if (c != ' ') {
                    throw new ParseException(String.format("Invalid character '%s' in column %s.", c, col),
                            levelReader.getLineNumber());
                }
            }
            // If the goal state line is shorter than the level width, we must check that no walls were omitted.
            for (; col < this.staticState.getNumCols(); ++col) {
                if (this.staticState.isWall(row, col)) {
                    throw new ParseException(
                            String.format("Goal state not matching initial state's wall on column %s.", col),
                            levelReader.getLineNumber());
                }
            }
            ++row;
        }

        this.staticState.setAgentGoals(agentGoals.stream().
                sorted(Comparator.comparingInt(Object::getId)).collect(Collectors.toList()));
        this.staticState.setBoxGoals(boxGoals.stream().
                sorted(Comparator.comparingInt(Object::getId)).collect(Collectors.toList()));

        return line;
    }

    private String parseClientNameSection(LineNumberReader levelReader)
            throws IOException, ParseException {
        String line = levelReader.readLine();
        if (line == null) {
            throw new ParseException("Expected a client name, but reached end of file.", levelReader.getLineNumber());
        }
        if (line.isBlank()) {
            throw new ParseException("Client name can not be blank.", levelReader.getLineNumber());
        }
        this.staticState.setClientName(line);
        line = levelReader.readLine();
        return line;
    }

    /*private String parseActionsSection(LineNumberReader levelReader)
            throws IOException, ParseException {
        Action[] jointAction = new Action[this.numAgents];

        while (true) {
            String line = levelReader.readLine();
            if (line == null) {
                throw new ParseException("Expected more action lines or end of actions section, but reached end of " +
                        "file.",
                        levelReader.getLineNumber());
            }

            if (line.length() > 0 && line.charAt(0) == '#') {
                return line;
            }

            String[] split = line.split(":");
            if (split.length < 1) {
                throw new ParseException("Invalid action line syntax - timestamp missing?",
                        levelReader.getLineNumber());
            }
            if (split.length > 2) {
                throw new ParseException("Invalid action line syntax - too many colons?", levelReader.getLineNumber());
            }

            // Parse action timestamp.
            long actionTime;
            try {
                actionTime = Long.valueOf(split[0]);
            } catch (NumberFormatException e) {
                throw new ParseException("Invalid action timestamp.", levelReader.getLineNumber());
            }

            // Parse and execute joint action.
            String[] actionsStr = split[1].split(";");
            if (actionsStr.length != this.numAgents) {
                throw new ParseException("Invalid number of agents in joint action.", levelReader.getLineNumber());
            }
            for (int i = 0; i < jointAction.length; ++i) {
                jointAction[i] = Action.parse(actionsStr[i]);
                if (jointAction[i] == null) {
                    throw new ParseException("Invalid joint action.", levelReader.getLineNumber());
                }
            }

            // Execute action.
            this.execute(jointAction, actionTime);
        }
    }

    private String parseSolvedSection(LineNumberReader levelReader)
            throws IOException, ParseException {
        String line = levelReader.readLine();
        if (line == null) {
            throw new ParseException("Expected a solved value, but reached end of file.", levelReader.getLineNumber());
        }

        if (!line.equals("true") && !line.equals("false")) {
            throw new ParseException("Invalid solved value.", levelReader.getLineNumber());
        }

        boolean logSolved = line.equals("true");
        boolean actuallySolved = true;
        for (int boxGoalId = 0; boxGoalId < this.numBoxGoals; ++boxGoalId) {
            short boxGoalRow = this.boxGoalRows[boxGoalId];
            short boxGoalCol = this.boxGoalCols[boxGoalId];
            byte boxGoalLetter = this.boxGoalLetters[boxGoalId];

            if (this.boxAt(boxGoalRow, boxGoalCol) != boxGoalLetter) {
                actuallySolved = false;
                break;
            }
        }
        State lastState = this.getState(this.numStates - 1);
        for (int agent = 0; agent < this.numAgents; ++agent) {
            short agentGoalRow = this.agentGoalRows[agent];
            short agentGoalCol = this.agentGoalCols[agent];
            short agentRow = lastState.agentRows[agent];
            short agentCol = lastState.agentCols[agent];

            if (agentGoalRow != -1 && (agentRow != agentGoalRow || agentCol != agentGoalCol)) {
                actuallySolved = false;
                break;
            }
        }

        if (logSolved && !actuallySolved) {
            throw new ParseException("Log summary claims level is solved, but the actions don't solve the level.",
                    levelReader.getLineNumber());
        } else if (!logSolved && actuallySolved) {
            throw new ParseException("Log summary claims level is not solved, but the actions solve the level.",
                    levelReader.getLineNumber());
        }

        line = levelReader.readLine();
        return line;
    }

    private String parseNumActionsSection(LineNumberReader levelReader)
            throws IOException, ParseException {
        String line = levelReader.readLine();
        if (line == null) {
            throw new ParseException("Expected a solved value, but reached end of file.", levelReader.getLineNumber());
        }

        long numActions;
        try {
            numActions = Long.valueOf(line);
        } catch (NumberFormatException e) {
            throw new ParseException("Invalid number of actions.", levelReader.getLineNumber());
        }

        if (numActions != this.numStates - 1) {
            throw new ParseException("Number of action does not conform to the number of actions in the sequence.",
                    levelReader.getLineNumber());
        }

        line = levelReader.readLine();
        return line;
    }

    private String parseTimeSection(LineNumberReader levelReader)
            throws IOException, ParseException {
        String line = levelReader.readLine();
        if (line == null) {
            throw new ParseException("Expected a solved value, but reached end of file.", levelReader.getLineNumber());
        }

        long lastStateTime;
        try {
            lastStateTime = Long.valueOf(line);
        } catch (NumberFormatException e) {
            throw new ParseException("Invalid time of last action.", levelReader.getLineNumber());
        }

        if (lastStateTime != this.getStateTime(this.numStates - 1)) {
            throw new ParseException("Last state time does not conform to the timestamp of the last action.",
                    levelReader.getLineNumber());
        }

        line = levelReader.readLine();
        return line;
    }*/

    private String parseEndSection(LineNumberReader levelReader)
            throws IOException, ParseException {
        return levelReader.readLine();
    }

    private String stripTrailingSpaces(String s) {
        int endIndex = s.length();
        while (endIndex > 0 && s.charAt(endIndex - 1) != '+') {
            --endIndex;
        }
        return s.substring(0, endIndex);
    }
}