package server;

import domain.Domain;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;

/**
 * TODO: Documentation.
 */
public class ArgumentParser {
    private boolean helpPrinted = false;

    public enum ServerInputMode {
        NONE,
        CLIENT,
        REPLAY
    }

    public enum ClientInputMode {
        NONE,
        FILE,
        DIRECTORY
    }

    public enum ServerOutputMode {
        NONE,
        GUI,
        LOG,
        BOTH
    }

    /**
     * Input and Output modes.
     * <p>
     * Valid combinations are:
     * CLIENT + FILE + {NONE, GUI, LOG, BOTH}
     * CLIENT + DIRECTORY + {NONE, LOG}
     * REPLAY + {NONE, GUI}
     * In total 8 combinations.
     */
    private ServerInputMode serverInputMode = ServerInputMode.NONE;
    private ClientInputMode clientInputMode = ClientInputMode.NONE;
    private ServerOutputMode serverOutputMode = ServerOutputMode.NONE;

    /**
     * Client options.
     */
    private String clientCommand = null;
    private Path levelPath = null;
    private int timeoutSeconds = 0;
    private Path logFilePath = null;

    /**
     * Replay options.
     */
    private Path[] replayFilePaths = null;

    /**
     * GUI options.
     */
    private int[] screens = null;
    private int msPerAction = 250;
    private boolean startPlaying = true;
    private boolean startFullscreen = false;
    private boolean startHiddenInterface = false;


    public ArgumentParser(String[] args) throws ArgumentException {
        if (args.length == 0) {
            ArgumentParser.printShortHelp();
            this.helpPrinted = true;
            return;
        }
        for (String s : args) {
            if (s.equals("-h")) {
                ArgumentParser.printLongHelp();
                this.helpPrinted = true;
                return;
            }
        }

        int i = 0;
        while (i < args.length) {
            switch (args[i]) {
                // GUI options.
                case "-g":
                    if (this.serverOutputMode == ServerOutputMode.NONE) {
                        this.serverOutputMode = ServerOutputMode.GUI;
                    } else if (this.serverOutputMode == ServerOutputMode.LOG) {
                        this.serverOutputMode = ServerOutputMode.BOTH;
                    }

                    ArrayList<Integer> screens = new ArrayList<>(8);
                    ++i;
                    while (i < args.length) {
                        if (args[i].charAt(0) == '-') {
                            --i;
                            break;
                        }
                        int screen;
                        try {
                            screen = Integer.parseInt(args[i]);
                        } catch (NumberFormatException e) {
                            --i;
                            break;
                        }
                        screens.add(screen);
                        ++i;
                    }
                    this.screens = screens.stream().mapToInt(s -> s).toArray();
                    break;

                case "-s":
                    ++i;
                    if (i >= args.length) {
                        throw new ArgumentException("Expected another argument after -s.");
                    }
                    try {
                        this.msPerAction = Integer.parseInt(args[i]);
                    } catch (NumberFormatException e) {
                        throw new ArgumentException("The argument after -s must be an integer.");
                    }
                    if (this.msPerAction < 0) {
                        throw new ArgumentException("The argument after -s can not be negative.");
                    }
                    break;

                case "-p":
                    this.startPlaying = false;
                    break;

                case "-f":
                    this.startFullscreen = true;
                    break;

                case "-i":
                    this.startHiddenInterface = true;
                    break;

                // Client options.
                case "-c":
                    if (this.serverInputMode == ServerInputMode.REPLAY) {
                        throw new ArgumentException("Can not use -c argument with -r.");
                    }
                    this.serverInputMode = ServerInputMode.CLIENT;

                    ++i;
                    if (i >= args.length) {
                        throw new ArgumentException("Expected another argument after -c.");
                    }
                    if (args[i].isBlank()) {
                        throw new ArgumentException("The argument after -c can not be blank.");
                    }
                    this.clientCommand = args[i];
                    break;

                case "-l":
                    if (this.serverInputMode == ServerInputMode.REPLAY) {
                        throw new ArgumentException("Can not use -l argument with -r.");
                    }
                    this.serverInputMode = ServerInputMode.CLIENT;

                    ++i;
                    if (i >= args.length) {
                        throw new ArgumentException("Expected another argument after -l.");
                    }
                    this.levelPath = Path.of(args[i]);
                    if (!Files.exists(this.levelPath) || !Files.isReadable(this.levelPath)) {
                        throw new ArgumentException("The level path may not exist, or has insufficient access.");
                    }
                    if (Files.isRegularFile(this.levelPath)) {
                        this.clientInputMode = ClientInputMode.FILE;
                    } else if (Files.isDirectory(this.levelPath)) {
                        this.clientInputMode = ClientInputMode.DIRECTORY;
                    } else {
                        throw new ArgumentException("Unknown level path type.");
                    }
                    break;

                case "-t":
                    if (this.serverInputMode == ServerInputMode.REPLAY) {
                        throw new ArgumentException("Can not use -t argument with -r.");
                    }
                    this.serverInputMode = ServerInputMode.CLIENT;

                    ++i;
                    if (i >= args.length) {
                        throw new ArgumentException("Expected another argument after -t.");
                    }
                    try {
                        this.timeoutSeconds = Integer.parseInt(args[i]);
                    } catch (NumberFormatException e) {
                        throw new ArgumentException("The argument after -t must be an integer.");
                    }
                    if (this.timeoutSeconds <= 0) {
                        throw new ArgumentException("The argument after -t must be positive.");
                    }
                    break;

                case "-o":
                    if (this.serverInputMode == ServerInputMode.REPLAY) {
                        throw new ArgumentException("Can not use -o argument with -r.");
                    }
                    this.serverInputMode = ServerInputMode.CLIENT;

                    if (this.serverOutputMode == ServerOutputMode.NONE) {
                        this.serverOutputMode = ServerOutputMode.LOG;
                    } else if (this.serverOutputMode == ServerOutputMode.GUI) {
                        this.serverOutputMode = ServerOutputMode.BOTH;
                    }

                    ++i;
                    if (i >= args.length) {
                        throw new ArgumentException("Expected another argument after -o.");
                    }
                    this.logFilePath = Path.of(args[i]);
                    if (this.logFilePath.getParent() != null && !Files.exists(this.logFilePath.getParent())) {
                        throw new ArgumentException(
                                "The parent directory of the log file must exist and have sufficient write access.");
                    }
                    if (!Files.notExists(this.logFilePath)) {
                        throw new ArgumentException("The log file may already exist, or has insufficient access.");
                    }
                    break;

                // Replay options.
                case "-r":
                    if (this.serverInputMode == ServerInputMode.CLIENT) {
                        throw new ArgumentException("Can not use -r argument with -c, -l, -t, or -o.");
                    }
                    this.serverInputMode = ServerInputMode.REPLAY;

                    ArrayList<Path> replayFilePaths = new ArrayList<>(8);
                    ++i;
                    while (i < args.length) {
                        if (args[i].charAt(0) == '-') {
                            --i;
                            break;
                        }
                        replayFilePaths.add(Path.of(args[i]));
                        ++i;
                    }
                    this.replayFilePaths = replayFilePaths.toArray(new Path[0]);
                    break;

                // Unknown argument.
                default:
                    throw new ArgumentException("Unknown argument: \"" + args[i] + "\".");
            }
            ++i;
        }

        // No GUI support for client with directory of levels.
        if (this.serverInputMode == ServerInputMode.CLIENT && this.clientInputMode == ClientInputMode.DIRECTORY &&
                (this.serverOutputMode == ServerOutputMode.GUI || this.serverOutputMode == ServerOutputMode.BOTH)) {
            throw new ArgumentException("GUI is not supported when running client on a directory of levels.");
        }
    }

    /**
     * Indicates whether the ArgumentParser printed a help message.
     * The server should exit if this returns true.
     */
    public boolean helpPrinted() {
        return this.helpPrinted;
    }

    /**
     * Server mode.
     */
    public ServerInputMode getServerInputMode() {
        return this.serverInputMode;
    }

    public ClientInputMode getClientInputMode() {
        return this.clientInputMode;
    }

    public ServerOutputMode getServerOutputMode() {
        return this.serverOutputMode;
    }

    public boolean hasGUIOutput() {
        return this.serverOutputMode == ServerOutputMode.GUI || this.serverOutputMode == ServerOutputMode.BOTH;
    }

    public boolean hasLogOutput() {
        return this.serverOutputMode == ServerOutputMode.LOG || this.serverOutputMode == ServerOutputMode.BOTH;
    }

    /**
     * Client options.
     */
    public String getClientCommand() {
        return this.clientCommand;
    }

    public Path getLevelPath() {
        return this.levelPath;
    }

    public int getTimeoutSeconds() {
        return this.timeoutSeconds;
    }

    public Path getLogFilePath() {
        return this.logFilePath;
    }

    /**
     * Replay options.
     */
    public Path[] getReplayFilePaths() {
        return this.replayFilePaths;
    }

    /**
     * GUI options.
     */
    public int[] getScreens() {
        return this.screens;
    }

    public int getMsPerAction() {
        return this.msPerAction;
    }

    public boolean getStartPlaying() {
        return this.startPlaying;
    }

    public boolean getStartFullscreen() {
        return this.startFullscreen;
    }

    public boolean getStartHiddenInterface() {
        return this.startHiddenInterface;
    }

    /**
     * Help messages.
     */
    @SuppressWarnings("LongLine")
    public static void printShortHelp() {
        String shortHelp = "" +
                "Show help message and exit:\n" +
                "    java -jar server.jar [-h]\n" +
                "Omitting the -h argument shows this abbreviated description.\n" +
                "Providing the -h argument shows a detailed description.\n" +
                "\n" +
                "Run a client on a level or a directory of levels, optionally output to GUI and/or log file:\n" +
                "    java -jar server.jar -c <client-cmd> -l <level-file-or-dir-path> [-t <seconds>]\n" +
                "                         [-g [<screen>] [-s <ms-per-action>] [-p] [-f] [-i]]\n" +
                "                         [-o <log-file-path>]\n" +
                "\n" +
                "Replay one or more log files, optionally output to synchronized GUIs:\n" +
                "    java -jar server.jar -r <log-file-path> [<log-file-path> ...]\n" +
                "                         [-g [<screen> ...] [-s <ms-per-action>] [-p] [-f] [-i]]";
        System.out.println(shortHelp);
    }

    @SuppressWarnings("LongLine")
    public static void printLongHelp() {
        String longHelp = "" +
                "The server modes are listed below.\n" +
                "The order of arguments is irrelevant.\n" +
                "Example invocations are given in the end.\n" +
                "\n" +
                "Show help message and exit:\n" +
                "    java -jar server.jar [-h]\n" +
                "Omitting the -h argument shows an abbreviated description.\n" +
                "Providing the -h argument shows this detailed description.\n" +
                "\n" +
                "Run a client on a level or a directory of levels, optionally output to GUI and/or log file:\n" +
                "    java -jar server.jar -c <client-cmd> -l <level-file-or-dir-path> [-t <seconds>]\n" +
                "                         [-g [<screen>] [-s <ms-per-action>] [-p] [-f] [-i]]\n" +
                "                         [-o <log-file-path>]\n" +
                "Where the arguments are as follows:\n" +
                "    -c <client-cmd>\n" +
                "        Specifies the command the server will use to start the client process, including all client arguments.\n" +
                "        The <client-cmd> string will be naïvely tokenized by splitting on whitespace, and\n" +
                "        then passed to your OS native process API through Java's ProcessBuilder.\n" +
                "    -l <level-file-or-dir-path>\n" +
                "        Specifies the path to either a single level file, or to a directory containing one or more level files.\n" +
                "    -t <seconds>\n" +
                "        Optional. Specifies a timeout in Seconds for the client.\n" +
                "        The server will terminate the client after this timeout.\n" +
                "        If this argument is not given, the server will never time the client out.\n" +
                "    -g [<screen>]\n" +
                "        Optional. Enables GUI output.\n" +
                "        The optional <screen> argument specifies which screen to start the GUI on.\n" +
                "        See notes on <screen> below for an explanation of the valid values.\n" +
                "    -s <ms-per-action>\n" +
                "        Optional. Set the GUI playback speed in milliseconds per action.\n" +
                "        By default the speed is 250 ms per action.\n" +
                "    -p\n" +
                "        Optional. Start the GUI paused.\n" +
                "        By default the GUI starts playing immediately.\n" +
                "    -f\n" +
                "        Optional. Start the GUI in fullscreen mode.\n" +
                "        By default the GUI starts in windowed mode.\n" +
                "    -i\n" +
                "        Optional. Start the GUI with interface hidden.\n" +
                "        By default the GUI shows interface elements for navigating playback.\n" +
                "    -o <log-file-path>\n" +
                "        Optional. Writes log file(s) of the client run.\n" +
                "        If the -l argument is a level file path, then a single log file is written to the given log file path.\n" +
                "        If the -l argument is a level directory path, then logs for the client run on all levels in the\n" +
                "        level directory are compressed as a zip file written to the given log file path.\n" +
                "        NB: The log file may *not* already exist (the server does not allow overwriting files).\n" +
                "\n" +
                "Replay one or more log files, optionally output to synchronized GUIs:\n" +
                "    java -jar server.jar -r <log-file-path> [<log-file-path> ...]\n" +
                "                         [-g [<screen> ...] [-s <ms-per-action>] [-p] [-f] [-i]]\n" +
                "Where the arguments are as follows:\n" +
                "    -r <log-file-path> [<log-file-path> ...]\n" +
                "        Specifies one or more log files to replay.\n" +
                "    -g [<screen> ...]\n" +
                "        Optional. Enables GUI output. The playback of the replays are synchronized.\n" +
                "        The optional <screen> arguments specify which screen to start the GUI on for each log file.\n" +
                "        See notes on <screen> below for an explanation of the valid values.\n" +
                "    -s <ms-per-action>\n" +
                "        Optional. Set the GUI playback speed in milliseconds per action.\n" +
                "        By default the speed is 250 ms per action.\n" +
                "    -p\n" +
                "        Optional. Start the GUI paused.\n" +
                "        By default the GUI starts playing immediately.\n" +
                "    -f\n" +
                "        Optional. Start the GUI in fullscreen mode.\n" +
                "        By default the GUI starts in windowed mode.\n" +
                "    -i\n" +
                "        Optional. Start the GUI with interface hidden.\n" +
                "        By default the GUI shows interface elements for navigating playback.\n" +
                "\n" +
                "Notes on the <screen> arguments:\n" +
                "    Values for the <screen> arguments are integers in the range 0..(<num-screens> - 1).\n" +
                "    The server attemps to enumerate screens from left-to-right, breaking ties with top-to-bottom.\n" +
                "    The real underlying screen ordering is system-defined, and the server may fail at enumerating in the above order.\n" +
                "    If no <screen> argument is given, then the 'default' screen is used, which is a system-defined screen.\n" +
                "\n" +
                "    E.g. in a grid aligned 2x2 screen setup, the server will attempt to enumerate the screens as:\n" +
                "    0: Top-left screen.\n" +
                "    1: Bottom-left screen.\n" +
                "    2: Top-right screen.\n" +
                "    3: Bottom-right screen.\n" +
                "\n" +
                "    E.g. in a 1x3 horizontally aligned screen setup, the server will attempt to enumerate the screens as:\n" +
                "    0: The left-most screen.\n" +
                "    1: The middle screen.\n" +
                "    2: The right-most screen.\n" +
                "\n" +
                "Supported domains (case-sensitive):\n" +
                "    " +
                String.join("\n    ", Domain.getSupportedDomains()) +
                "\n\n" +
                "Client example invocations:\n" +
                "    # Client on single level, no output.\n" +
                "    java -jar server.jar -c \"java ExampleClient\" -l \"levels/example.lvl\"\n" +
                "\n" +
                "    # Client on single level, output to GUI on default screen.\n" +
                "    java -jar server.jar -c \"java ExampleClient\" -l \"levels/example.lvl\" -g\n" +
                "\n" +
                "    # Client on single level, output to GUI on screen 0.\n" +
                "    java -jar server.jar -c \"java ExampleClient\" -l \"levels/example.lvl\" -g 0\n" +
                "\n" +
                "    # Client on single level, output to log file.\n" +
                "    java -jar server.jar -c \"java ExampleClient\" -l \"levels/example.lvl\" -o \"logs/example.log\"\n" +
                "\n" +
                "    # Client on single level, output to GUI on default screen and to log file.\n" +
                "    java -jar server.jar -c \"java ExampleClient\" -l \"levels/example.lvl\" -g -o \"logs/example.log\"\n" +
                "\n" +
                "    # Client on a directory of levels, no output.\n" +
                "    java -jar server.jar -c \"java ExampleClient\" -l \"levels\"\n" +
                "\n" +
                "    # Client on a directory of levels, output to log archive.\n" +
                "    java -jar server.jar -c \"java ExampleClient\" -l \"levels\" -o \"logs.zip\"\n" +
                "\n" +
                "Replay example invocations:\n" +
                "    # Replay a single log file, no output.\n" +
                "    java -jar server.jar -r \"logs/example.log\"\n" +
                "\n" +
                "    # Replay a single log file, output to GUI on default screen.\n" +
                "    java -jar server.jar -r \"logs/example.log\" -g\n" +
                "\n" +
                "    # Replay two log files, output to synchronized GUIs on screen 0 and 1.\n" +
                "    # Start the GUIs paused, in fullscreen mode and with hidden interface elements to avoid spoilers.\n" +
                "    # Play back actions at a speed of one action every 500 milliseconds.\n" +
                "    java -jar server.jar -r \"logs/example1.log\" \"logs/example2.log\" -g 0 1 -p -f -i -s 500";
        System.out.println(longHelp);
    }
}

