package server;

import client.Client;
import client.Timeout;
import domain.Domain;
import domain.ParseException;
import gui.PlaybackManager;

import javax.swing.*;
import java.awt.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/**
 * TODO: Clean shutdown of server on Ctrl-C, SIGINT, or whatever mechanisms that cause the JVM to shut down nicely.
 * <p>
 * TODO NICE-HAVE:
 * - DOMAIN: More implementations.
 * - GUI: More speed presets?
 * - GUI: Help overlay.
 * - GUI: Reloading/navigating new client/domains without server restart?
 * <p>
 * <p>
 * <p>
 * TEST ARGUMENTS
 * -c "java src/TestClient.java" -l "./levels/MATest.lvl" -t 10 -o "./logs/MATest1.log"
 * -c "java src/TestClient.java" -l "./levels/MATest.lvl" -t 10 -o "./logs/MATest2.log"
 * -r "./logs/MATest1.log" "./logs/MATest2.log" -g 0 1
 */
public class Server {
    public static final boolean PRINT_DEBUG = "true".equalsIgnoreCase(System.getenv("AIMAS_SERVER_DEBUG"));

    // FIXME: Context for clean shutdown on shutdown hook.
//    private static final Thread shutdownThread = new Thread(Server::shutdownServer);
//    private static volatile boolean shutdown = false;
//    private static volatile boolean shutdownDone = false;
//    private static volatile Timeout currentTimeout;
//    private static volatile PlaybackManager currentPlaybackManager;

    public static void main(String[] args) {
        Thread.currentThread().setName("MainThread");
        Server.printDebug("Thread started.");

        // FIXME: Hotfix for UI scaling.
        System.setProperty("sun.java2d.uiScale.enabled", "false");

        // FIXME:
//        Runtime.getRuntime().addShutdownHook(Server.shutdownThread);

        ArgumentParser arguments;
        try {
            arguments = new ArgumentParser(args);
        } catch (ArgumentException e) {
            Server.printError(e.getMessage());
            return;
        }

        if (arguments.helpPrinted()) {
            return;
        }

        switch (arguments.getServerInputMode()) {
            case NONE:
                Server.printError("No client or replay files specified.");
                return;
            case CLIENT:
                switch (arguments.getClientInputMode()) {
                    case NONE:
                        Server.printError("No level path given.");
                        return;
                    case FILE:
                        runClientOnSingleLevel(arguments);
                        break;
                    case DIRECTORY:
                        runClientOnLevelDirectory(arguments);
                        break;
                    // TODO: Zip?
                }
                break;
            case REPLAY:
                runReplays(arguments);
                break;
        }

        // TODO: Loop Main thread on System.gc() at fixed frequency? As a workaround to the reallocation of buffers
        //       in the DomainPanels not getting garbage collected at any reasonable pace.
        //       Currently .gc() is called by the EDT during DomainPanel.validateBuffers(). This may also be
        //       reasonable when we mostly don't reallocate the buffers. We could also compound that .gc() if we
        //       expect we may have several times when there are multiple windows reallocating buffers at the same time.

        Server.printDebug("Thread shut down.");

        // FIXME:
//        synchronized (Server.class)
//        {
//            if (!Server.shutdown) {
//                // No need for shutdown hook when we're shutting down normally.
//                Runtime.getRuntime().removeShutdownHook(Server.shutdownThread);
//            }
//            Server.shutdownDone = true;
//            Server.class.notifyAll();
//        }
    }

    private static void runClientOnSingleLevel(ArgumentParser args) {
        Server.printInfo(String.format("Running client on level: %s", args.getLevelPath()));

        // Load domain.
        Domain domain;
        try {
            domain = Domain.loadLevel(args.getLevelPath());
        } catch (ParseException e) {
            // TODO: Better error message (level invalid, rather than "failing" to parse).
            Server.printError("Could not load domain, failed to parse level file.");
            Server.printError(e.getMessage());
            return;
        } catch (IOException e) {
            Server.printError("IOException while loading domain.");
            Server.printError(e.getMessage());
            return;
        }

        // Load GUI.
        PlaybackManager playbackManager = null;
        if (args.hasGUIOutput()) {
            Server.printDebug("Loading GUI.");
            try {
                UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
            } catch (ClassNotFoundException |
                    UnsupportedLookAndFeelException |
                    IllegalAccessException |
                    InstantiationException e) {
                Server.printWarning("Could not set system look and feel.");
                Server.printWarning(e.getMessage());
            }
            var gcs = getGraphicsConfigurations(1, args.getScreens());
            var domains = new Domain[]{domain};
            playbackManager = new PlaybackManager(domains, gcs);
        } else {
            // The domain can discard past states if we don't need to be able to seek states in the GUI.
            domain.allowDiscardingPastStates();
        }

        // Open log file.
        OutputStream logFileStream;
        if (args.hasLogOutput()) {
            try {
                logFileStream = Files.newOutputStream(args.getLogFilePath(),
                        StandardOpenOption.CREATE_NEW,
                        StandardOpenOption.WRITE);
            } catch (IOException e) {
                Server.printError("Could not create log file: " + args.getLogFilePath());
                Server.printError(e.getMessage());
                return;
            }
        } else {
            logFileStream = OutputStream.nullOutputStream();
        }

        // Load and start client.
        Client client;
        Timeout timeout = new Timeout();
        try {
            long timeoutNS = args.getTimeoutSeconds() * 1_000_000_000L;
            client = new Client(domain, args.getClientCommand(), logFileStream, true, timeout, timeoutNS);
        } catch (Exception e) {
            // TODO: Start writing errors to log file also? Will complicate/break parsing from domains though.
            Server.printError("Could not start client process.");
            Server.printError(e.getMessage());
            try {
                logFileStream.close();
            } catch (IOException e1) {
                Client.printError("Could not close log file.");
                Client.printError(e1.getMessage());
            }
            return;
        }

        // Start GUI.
        if (args.hasGUIOutput()) {
            assert playbackManager != null;
            // FIXME: This may hang if the EDT crashed after the constructor (or during this call)?
            Server.printDebug("Starting GUI.");
            playbackManager.startGUI(args.getStartFullscreen(),
                    args.getStartHiddenInterface(),
                    args.getMsPerAction(),
                    args.getStartPlaying());
            playbackManager.focusPlaybackFrame(0);
        }

        // Start client protocol.
        client.startProtocol();

        // Wait for GUI to shut down.
        if (args.hasGUIOutput()) {
            assert playbackManager != null;
            // FIXME: Holding class sync lock will not allow shutdown hook to signal GUI shutdown, deadlocking.
            playbackManager.waitShutdown();
            timeout.expire(); // Does nothing if timeout already expired or stopped.
        }

        // Wait for client to shut down (if it hasn't already while GUI ran).
        // FIXME: Join not interruptible for clean shutdown.
        client.waitShutdown();
    }

    private static void runClientOnLevelDirectory(ArgumentParser args) {
        var levelFileFilter = new DirectoryStream.Filter<Path>() {
            @Override
            public boolean accept(Path entry) {
                return Files.isReadable(entry) &&
                        Files.isRegularFile(entry) &&
                        entry.getFileName().toString().endsWith(".lvl") &&
                        entry.getFileName().toString().length() > 4;
            }
        };

        try (var levelDirectory = Files.newDirectoryStream(args.getLevelPath(), levelFileFilter)) {
            // Open log file.
            OutputStream logFileStream;
            ZipOutputStream logZipStream;
            if (args.hasLogOutput()) {
                try {
                    logZipStream = new ZipOutputStream(new BufferedOutputStream(Files.newOutputStream(
                            args.getLogFilePath(),
                            StandardOpenOption.CREATE_NEW,
                            StandardOpenOption.WRITE)));
                    logFileStream = logZipStream;
                } catch (IOException e) {
                    Server.printError("Could not create log file: " + args.getLogFilePath());
                    Server.printError(e.getMessage());
                    return;
                }
            } else {
                logFileStream = OutputStream.nullOutputStream();
                logZipStream = new ZipOutputStream(logFileStream);
            }

            ArrayList<String> levelNames = new ArrayList<>();
            ArrayList<String[]> levelStatus = new ArrayList<>();

            for (Path levelPath : levelDirectory) {
                Server.printInfo(String.format("Running client on level file: %s", levelPath.toString()));

                // Load domain.
                Domain domain;
                try {
                    domain = Domain.loadLevel(levelPath);
                } catch (ParseException e) {
                    Server.printError("Could not load domain, failed to parse level file.");
                    Server.printError(e.getMessage());
                    continue;
                } catch (IOException e) {
                    Server.printError("IOException while loading domain.");
                    e.printStackTrace();
                    continue;
                }

                // Never run with GUI, always discard states.
                domain.allowDiscardingPastStates();

                // Prepare next log entry.
                String levelFileName = levelPath.getFileName().toString();
                String logEntryName = levelFileName.substring(0, levelFileName.length() - 4) + ".log";
                try {
                    logZipStream.putNextEntry(new ZipEntry(logEntryName));
                } catch (IOException e) {
                    Server.printError("Could not create log file entry for level.");
                    Server.printError(e.getMessage());
                    continue;
                }

                // Load and start client.
                Client client;
                Timeout timeout = new Timeout();
                try {
                    long timeoutNS = args.getTimeoutSeconds() * 1_000_000_000L;
                    client = new Client(domain, args.getClientCommand(), logFileStream, false, timeout, timeoutNS);
                } catch (Exception e) {
                    // TODO: Start writing errors to log file also? Will complicate/break parsing from domains though.
                    Server.printError("Could not start client process.");
                    Server.printError(e.getMessage());
                    continue;
                }

                // Start client protocol.
                client.startProtocol();

                // Wait for client to shut down.
                client.waitShutdown();

                // Aggregate level summaries.
                levelNames.add(domain.getLevelName());
                levelStatus.add(domain.getStatus());

                // Clear up resources and wait a moment before we proceed to next level.
                domain = null;
                timeout = null;
                client = null;
                System.gc();
            }

            // Write summary to log file.
            try {
                logZipStream.putNextEntry(new ZipEntry("summary.txt"));
                BufferedWriter logWriter
                        = new BufferedWriter(new OutputStreamWriter(logFileStream,
                        StandardCharsets.US_ASCII.newEncoder()));
                for (int i = 0; i < levelNames.size(); ++i) {
                    logWriter.write("Level name: ");
                    logWriter.write(levelNames.get(i));
                    logWriter.newLine();
                    for (String statusLine : levelStatus.get(i)) {
                        logWriter.write(statusLine);
                        logWriter.newLine();
                    }
                    logWriter.newLine();
                    logWriter.flush();
                }
            } catch (IOException e) {
                Server.printError("Could not write summary to log file.");
                Server.printError(e.getMessage());
            }

            // Close log file.
            try {
                logZipStream.close();
            } catch (IOException e) {
                Server.printError("Could not close log file.");
                Server.printError(e.getMessage());
                return;
            }
        } catch (IOException e) {
            Server.printError("Could not open levels directory.");
            Server.printError(e.getMessage());
            return;
        }
    }

    private static void runReplays(ArgumentParser args) {
        // Load domains.
        Path[] replayFilePaths = args.getReplayFilePaths();
        Domain[] domains = new Domain[replayFilePaths.length];
        for (int i = 0; i < replayFilePaths.length; i++) {
            try {
                Server.printInfo(String.format("Loading log file: %s", replayFilePaths[i]));
                domains[i] = Domain.loadReplay(replayFilePaths[i]);
            } catch (ParseException e) {
                // TODO: Better error message (level invalid, rather than "failing" to parse).
                Server.printError("Could not load domain, failed to parse log file.");
                Server.printError(e.getMessage());
                return;
            } catch (IOException e) {
                Server.printError("IOException while loading domain.");
                Server.printError(e.getMessage());
                return;
            }
        }

        PlaybackManager playbackManager = null;
        if (args.hasGUIOutput()) {
            Server.printDebug("Loading GUI.");
            try {
                UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
            } catch (ClassNotFoundException |
                    UnsupportedLookAndFeelException |
                    IllegalAccessException |
                    InstantiationException e) {
                Server.printWarning("Could not set system look and feel.");
                Server.printWarning(e.getMessage());
            }
            var gcs = getGraphicsConfigurations(domains.length, args.getScreens());
            playbackManager = new PlaybackManager(domains, gcs);

            Server.printDebug("Starting GUI.");
            playbackManager.startGUI(args.getStartFullscreen(),
                    args.getStartHiddenInterface(),
                    args.getMsPerAction(),
                    args.getStartPlaying());
            playbackManager.focusPlaybackFrame(0);

            playbackManager.waitShutdown();
        }
    }

//    private static void shutdownServer()
//    {
//        /* FIXME: Function synchronized?
//
//           TODO: Graceful shutdown. Stuff that needs to happen:
//                 - Time out client process.
//                 - Close GUI.
//                 - Close log file.
//                 - Join on main thread, which in turn has joined on the other threads.
//
//           FIXME: Confirm that JVM runs System.exit() on SIGINT/SIGTERM/etc, so
//                  we're sure that non-daemon threads are still executing. Otherwise,
//                  we can't join on main, etc.
//         */
//        Server.printDebug("Shutdown begin");
//
//        // Signal shutdown.
//        Server.currentTimeout.expire();
//        if (Server.currentPlaybackManager != null)
//        {
//            Server.currentPlaybackManager.shutdownGUI();
//        }
//
//        // Wait until shutdown done.
//        synchronized (Server.class)
//        {
//            while (!Server.shutdownDone)
//            {
//                try
//                {
//                    Server.class.wait();
//                }
//                catch (InterruptedException ignored)
//                {
//                }
//            }
//        }
//
//        Server.printDebug("Shutdown end");
//    }

    /**
     * Returns a GraphicsConfiguration array of size numScreens, attempting to map the user-specified
     * screen numbers (if provided) to the GraphicsDevices corresponding to a logical
     * left-to-right, top-to-bottom enumeration. See the screen notes in ArgumentParser.
     */
    @SuppressWarnings("SameParameterValue")
    private static GraphicsConfiguration[] getGraphicsConfigurations(int numScreens, int[] screens) {
        final GraphicsConfiguration defaultScreen = GraphicsEnvironment
                .getLocalGraphicsEnvironment()
                .getDefaultScreenDevice()
                .getDefaultConfiguration();

        // WARNING! getScreenDevices() gives a direct reference to underlying array.
        // Copy before sorting, so we don't mutate underlying array.
        GraphicsDevice[] gds = GraphicsEnvironment.getLocalGraphicsEnvironment().getScreenDevices();
        gds = Arrays.copyOf(gds, gds.length);
        // Attempt sorting for the logical enumeration. May not work, e.g. if bounds are not relative to each other.
        Arrays.sort(gds,
                Comparator.comparingInt((GraphicsDevice gd) -> gd.getDefaultConfiguration().getBounds().x)
                        .thenComparingInt(gd -> gd.getDefaultConfiguration().getBounds().y));

        var gcs = new GraphicsConfiguration[numScreens];
        for (int i = 0; i < numScreens; ++i) {
            if (i >= screens.length) {
                // Unspecified.
                gcs[i] = defaultScreen;
            } else if (screens[i] < 0 || screens[i] >= gds.length) {
                // Out of range.
                gcs[i] = defaultScreen;
                Server.printWarning("No screen #" + screens[i] + "; using default screen.");
            } else {
                // User-specified.
                gcs[i] = gds[screens[i]].getDefaultConfiguration();
            }
        }

        return gcs;
    }

    public static void printTodo(String msg) {
        if (!Server.PRINT_DEBUG) {
            return;
        }
        synchronized (System.out) {
            System.out.print("[server][todo] ");
            System.out.println(msg);
        }
    }

    public static void printDebug(String msg) {
        if (!Server.PRINT_DEBUG) {
            return;
        }
        synchronized (System.out) {
            System.out.print("[server][debug]");
            System.out.print("[" + Thread.currentThread().getName() + "] ");
            System.out.println(msg);
        }
    }

    public static void printInfo(String msg) {
        synchronized (System.out) {
            System.out.print("[server][info] ");
            System.out.println(msg);
        }
    }

    public static void printWarning(String msg) {
        synchronized (System.out) {
            System.out.print("[server][warning] ");
            System.out.println(msg);
        }
    }

    public static void printError(String msg) {
        synchronized (System.out) {
            System.out.print("[server][error] ");
            System.out.println(msg);
        }
    }
}
