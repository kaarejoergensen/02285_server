package domain.gridworld.hospital2.state.parser;

import domain.ParseException;
import domain.gridworld.hospital2.state.State;
import domain.gridworld.hospital2.state.objects.Object;
import domain.gridworld.hospital2.state.objects.*;
import lombok.Getter;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import shared.Action;
import shared.Farge;

import java.awt.*;
import java.io.IOException;
import java.nio.charset.MalformedInputException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class StateParser {
    private Logger serverLogger = LogManager.getLogger("server");

    private Path domainFile;
    private boolean isReplay;
    private HashMap<String, Color> agentColors;
    private HashMap<String, Color> boxColors;

    @Getter private StaticState staticState;
    @Getter private State state;

    @Getter private List<Action[]> actions;
    @Getter private List<Long> actionTimes;
    @Getter private boolean logSolved;

    @Getter String level;

    public StateParser(Path domainFile, boolean isReplay) {
        this.domainFile = domainFile;
        this.isReplay = isReplay;

        this.agentColors = new HashMap<>();
        this.boxColors = new HashMap<>();

        this.staticState = new StaticState();
    }

    public void parse() throws IOException, ParseException {
        var tStart = System.nanoTime();
        try (LevelReader levelReader = new LevelReader(domainFile)) {
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

                checkObjectsEnclosedInWalls();

                if (!line.stripTrailing().equalsIgnoreCase("#end")) {
                    throw new ParseException("Expected end section (#end).", levelReader.getLineNumber());
                }
                line = this.parseEndSection(levelReader);

                // If this is a log file, then parse additional sections.
                if (isReplay) {
                    // Parse client name.
                    if (!line.stripTrailing().equalsIgnoreCase("#clientname")) {
                        throw new ParseException("Expected client name section (#clientname).",
                                levelReader.getLineNumber());
                    }
                    line = this.parseClientNameSection(levelReader);

                    // Parse and simulate actions.
                    if (!line.stripTrailing().equalsIgnoreCase("#actions")) {
                        throw new ParseException("Expected actions section (#actions).", levelReader.getLineNumber());
                    }
                    line = this.parseActionsSection(levelReader);

                    if (!line.stripTrailing().equalsIgnoreCase("#end")) {
                        throw new ParseException("Expected end section (#end).", levelReader.getLineNumber());
                    }
                    line = this.parseEndSection(levelReader);

                    // Parse summary to check if it is consistent with simulation.
                    if (!line.stripTrailing().equalsIgnoreCase("#solved")) {
                        throw new ParseException("Expected solved section (#solved).", levelReader.getLineNumber());
                    }
                    line = this.parseSolvedSection(levelReader);

                    if (!line.stripTrailing().equalsIgnoreCase("#numactions")) {
                        throw new ParseException("Expected numactions section (#numactions).", levelReader.getLineNumber());
                    }
                    line = this.parseNumActionsSection(levelReader);

                    if (!line.stripTrailing().equalsIgnoreCase("#time")) {
                        throw new ParseException("Expected time section (#time).", levelReader.getLineNumber());
                    }
                    line = this.parseTimeSection(levelReader);

                    if (!line.stripTrailing().equalsIgnoreCase("#end")) {
                        throw new ParseException("Expected end section (#end).", levelReader.getLineNumber());
                    }
                    line = this.parseEndSection(levelReader);
                }

                if (line != null) {
                    throw new ParseException("Expected no more content after end section.",
                            levelReader.getLineNumber());
                }
                this.level = levelReader.getLevel();
            } catch (MalformedInputException e) {
                throw new ParseException("Level file content not valid ASCII.", levelReader.getLineNumber());
            }
        }
        var tEnd = System.nanoTime();
        serverLogger.debug(String.format("Parsing time: %.3f ms.", (tEnd - tStart) / 1_000_000_000.0));
    }

    private String parseNameSection(LevelReader levelReader)
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

    private String parseColorsSection(LevelReader levelReader)
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
            Farge farge = Farge.fromString(colorName);
            if (farge == null) {
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
                    String id = "A" + (s - '0');
                    if (this.agentColors.containsKey(id)) {
                        throw new ParseException(String.format("Agent '%s' already has a color specified.", s),
                                levelReader.getLineNumber());
                    }
                    this.agentColors.put(id, farge.color);
                } else if ('A' <= s && s <= 'Z') {
                    String id = "B" + (s - 'A');
                    if (this.boxColors.containsKey(id)) {
                        throw new ParseException(String.format("Box '%s' already has a color specified.", s),
                                levelReader.getLineNumber());
                    }
                    this.boxColors.put(id, farge.color);
                } else {
                    throw new ParseException(String.format("Invalid agent or box symbol: '%s'.", s),
                            levelReader.getLineNumber());
                }
            }
        }
    }

    private String parseInitialSection(LevelReader levelReader)
            throws IOException, ParseException {
        short numRows = 0;
        List<List<Boolean>> map = new ArrayList<>();
        List<Box> boxes = new ArrayList<>();
        List<Agent> agents = new ArrayList<>(10);

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
                    String id = "A" + c;
                    if (agents.stream().anyMatch(a -> a.getId().equals(id))) {
                        throw new ParseException(
                                String.format("Agent '%s' appears multiple times in initial state.", c),
                                levelReader.getLineNumber());
                    }
                    Color agentColor = this.agentColors.get(id);
                    if (agentColor == null) {
                        throw new ParseException(String.format("Agent '%s' has no color specified.", c),
                                levelReader.getLineNumber());
                    }
                    agents.add(new Agent(id, c, new Coordinate(numRows, col), agentColor));
                } else if ('A' <= c && c <= 'Z') {
                    // Box.
                    String id = "B" + c + "" + numRows + "" + col;
                    Color boxColor = this.boxColors.get("B" + (c - 'A'));
                    if (boxColor == null) {
                        throw new ParseException(String.format("Box '%s' has no color specified.", c),
                                levelReader.getLineNumber());
                    }
                    boxes.add(new Box(id, c, new Coordinate(numRows, col), boxColor));
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
        this.staticState.setMap(new Map(map));
        this.staticState.setNumAgents((byte) agents.size());

        var boxMap = boxes.stream().collect(Collectors.toMap(Box::getCoordinate, Function.identity()));
        var agentMap = agents.stream().collect(Collectors.toMap(Agent::getCoordinate, Function.identity()));
        this.state = new State(agentMap, boxMap);

        if (agents.isEmpty()) {
            throw new ParseException("Level contains no agents.", levelReader.getLineNumber());
        }

        return line;
    }

    private String parseGoalSection(LevelReader levelReader)
            throws IOException, ParseException {
        short row = 0;

        List<Goal> agentGoals = new ArrayList<>(this.state.getAgents().size());
        List<Goal> boxGoals = new ArrayList<>(this.state.getBoxes().size());

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
                Coordinate coordinate = new Coordinate(row, col);
                if (c == '+') {
                    // Wall.
                    if (!this.staticState.getMap().isWall(coordinate)) {
                        // Which doesn't match a wall in the initial state.
                        throw new ParseException(
                                String.format("Initial state has no wall at column %d, but goal state does.", col),
                                levelReader.getLineNumber());
                    }
                } else if (this.staticState.getMap().isWall(coordinate)) {
                    // Missing wall compared to the initial state.
                    throw new ParseException(
                            String.format("Goal state not matching initial state's wall on column %d.", col),
                            levelReader.getLineNumber());
                } else if ('0' <= c && c <= '9') {
                    // Agent.
                    short number = (short) (c - '0');
                    if (number >= this.state.getAgents().size()) {
                        throw new ParseException(
                                String.format("Goal state has agent '%s' who does not appear in the initial state.", c),
                                levelReader.getLineNumber());
                    }
                    String id = "GA" + number;
                    if (agentGoals.stream().anyMatch(g -> g.getId().equals(id))) {
                        throw new ParseException(String.format("Agent '%s' appears multiple times in goal state.", c),
                                levelReader.getLineNumber());
                    }
                    agentGoals.add(new Goal(id, c, coordinate, null));
                } else if ('A' <= c && c <= 'Z') {
                    // Box.
                    String id = "BA" + (c - 'A') + row + "" + col;
                    boxGoals.add(new Goal(id, c, coordinate, null));
                } else if (c != ' ') {
                    throw new ParseException(String.format("Invalid character '%s' in column %s.", c, col),
                            levelReader.getLineNumber());
                }
            }
            // If the goal state line is shorter than the level width, we must check that no walls were omitted.
            for (; col < this.staticState.getNumCols(); ++col) {
                if (this.staticState.getMap().isWall(new Coordinate(row, col))) {
                    throw new ParseException(
                            String.format("Goal state not matching initial state's wall on column %s.", col),
                            levelReader.getLineNumber());
                }
            }
            ++row;
        }

        this.staticState.setAgentGoals(agentGoals);
        this.staticState.setBoxGoals(boxGoals);
        this.staticState.setAllGoals(Stream.concat(agentGoals.stream(), boxGoals.stream()).collect(Collectors.toList()));

        return line;
    }

    private String parseClientNameSection(LevelReader levelReader)
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

    //TODO: Implement properly
    private void checkObjectsEnclosedInWalls() throws ParseException {
        Predicate<Object> pred = o -> !staticState.getMap().isCell(o.getCoordinate());
        if (this.state.getAgents().stream().anyMatch(pred)
                || this.state.getBoxes().stream().anyMatch(pred)
                || this.staticState.getAgentGoals().stream().anyMatch(pred)
                || this.staticState.getBoxGoals().stream().anyMatch(pred)) {
            throw new ParseException("One or more objects not enclosed in walls");
        }
    }

    private String parseActionsSection(LevelReader levelReader)
            throws IOException, ParseException {
        this.actions = new ArrayList<>();
        this.actionTimes = new ArrayList<>();

        while (true) {
            Action[] jointAction = new Action[this.state.getAgents().size()];
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
                actionTime = Long.parseLong(split[0]);
            } catch (NumberFormatException e) {
                throw new ParseException("Invalid action timestamp.", levelReader.getLineNumber());
            }

            // Parse and execute joint action.
            String[] actionsStr = split[1].split(";");
            if (actionsStr.length != this.state.getAgents().size()) {
                throw new ParseException("Invalid number of agents in joint action.", levelReader.getLineNumber());
            }
            for (int i = 0; i < jointAction.length; ++i) {
                jointAction[i] = Action.parse(actionsStr[i]);
                if (jointAction[i] == null) {
                    throw new ParseException("Invalid joint action.", levelReader.getLineNumber());
                }
            }

            // Execute action.
            this.actions.add(jointAction);
            this.actionTimes.add(actionTime);
        }
    }

    private String parseSolvedSection(LevelReader levelReader)
            throws IOException, ParseException {
        String line = levelReader.readLine();
        if (line == null) {
            throw new ParseException("Expected a solved value, but reached end of file.", levelReader.getLineNumber());
        }

        if (!line.equals("true") && !line.equals("false")) {
            throw new ParseException("Invalid solved value.", levelReader.getLineNumber());
        }

        this.logSolved = line.equals("true");

        line = levelReader.readLine();
        return line;
    }

    private String parseNumActionsSection(LevelReader levelReader)
            throws IOException, ParseException {
        String line = levelReader.readLine();
        if (line == null) {
            throw new ParseException("Expected a solved value, but reached end of file.", levelReader.getLineNumber());
        }

        long numActions;
        try {
            numActions = Long.parseLong(line);
        } catch (NumberFormatException e) {
            throw new ParseException("Invalid number of actions.", levelReader.getLineNumber());
        }

        if (numActions != this.actions.size()) {
            throw new ParseException("Number of action does not conform to the number of actions in the sequence.",
                    levelReader.getLineNumber());
        }

        line = levelReader.readLine();
        return line;
    }

    private String parseTimeSection(LevelReader levelReader)
            throws IOException, ParseException {
        String line = levelReader.readLine();
        if (line == null) {
            throw new ParseException("Expected a solved value, but reached end of file.", levelReader.getLineNumber());
        }

        long lastStateTime;
        try {
            lastStateTime = Long.parseLong(line);
        } catch (NumberFormatException e) {
            throw new ParseException("Invalid time of last action.", levelReader.getLineNumber());
        }
        if (lastStateTime != this.actionTimes.get(this.actionTimes.size() - 1)) {
            throw new ParseException("Last state time does not conform to the timestamp of the last action.",
                    levelReader.getLineNumber());
        }

        line = levelReader.readLine();
        return line;
    }

    private String parseEndSection(LevelReader levelReader)
            throws IOException {
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