package server;

import client.Client;
import client.Timeout;
import domain.Domain;
import domain.ParseException;
import domain.gridworld.hospital2.Hospital2Domain;
import gui.PlaybackManager;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import javax.swing.*;
import java.awt.*;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
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

public class Server {
    private Logger serverLogger = LogManager.getLogger("server");
    private Logger clientLogger = LogManager.getLogger("client");

    private ArgumentParser arguments;

    // FIXME: Context for clean shutdown on shutdown hook.
//    private static final Thread shutdownThread = new Thread(Server::shutdownServer);
//    private static volatile boolean shutdown = false;
//    private static volatile boolean shutdownDone = false;
//    private static volatile Timeout currentTimeout;
//    private static volatile PlaybackManager currentPlaybackManager;

    public static void main(String[] args) {
        Server server = new Server(args);
        server.run();
    }

    private Server(String[] args) {
        Thread.currentThread().setName("MainThread");
        serverLogger.debug("Thread started.");

        // FIXME: Hotfix for UI scaling.
        System.setProperty("sun.java2d.uiScale.enabled", "false");

        // FIXME:
        //Runtime.getRuntime().addShutdownHook(Server.shutdownThread);

        try {
            this.arguments = new ArgumentParser(args);
        } catch (IllegalArgumentException e) {
            serverLogger.error(e.getMessage(), e);
            System.exit(-1);
        }

        if (this.arguments.isHelpPrinted()) {
            System.exit(0);
        }
    }

    public void run() {
        switch (arguments.getServerInputMode()) {
            case NONE:
                serverLogger.error("No client or replay files specified.");
                return;
            case CLIENT:
                switch (arguments.getClientInputMode()) {
                    case NONE:
                        serverLogger.error("No level path given.");
                        return;
                    case FILE:
                        this.runClientOnSingleLevel(arguments);
                        break;
                    case DIRECTORY:
                        this.runClientOnLevelDirectory(arguments);
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

        serverLogger.debug("Thread shut down.");

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

    private void runClientOnSingleLevel(ArgumentParser args) {
        OutputStream logFileStream = this.createLogFileStream(args.hasLogOutput(), args.getLogFilePath());

        this.runLevel(args.getLevelPath(), args, logFileStream, args.hasGUIOutput(), true);
    }

    private void runClientOnLevelDirectory(ArgumentParser args) {
        try (var levelDirectory = Files.newDirectoryStream(args.getLevelPath(), createLevelFilter())) {
            ZipOutputStream logZipStream = new ZipOutputStream(this.createLogFileStream(args.hasLogOutput(), args.getLogFilePath()));

            ArrayList<Domain> domains = new ArrayList<>();

            for (Path levelPath : levelDirectory) {
                this.createZipEntry(levelPath, logZipStream);

                Domain domain = this.runLevel(levelPath, args, logZipStream, false, false);
                if (domain == null) continue;

                domains.add(domain);

                System.gc();
            }

            this.writeSummary(logZipStream, domains);

            try {
                logZipStream.close();
            } catch (IOException e) {
                serverLogger.error("Could not close log file. " + e.getMessage(), e);
            }
        } catch (IOException e) {
            serverLogger.error("Could not open levels directory. " + e.getMessage(), e);
        }
    }

    private void runReplays(ArgumentParser args) {
        Domain[] domains = this.loadReplayDomains(args.getReplayFilePaths());

        if (args.hasGUIOutput()) {
            PlaybackManager playbackManager = this.loadAndStartGUI(domains, args);
            playbackManager.waitShutdown();
        }
    }

    private OutputStream createLogFileStream(boolean hasLogOutput, Path logFilePath) {
        if (hasLogOutput) {
            try {
                return Files.newOutputStream(logFilePath, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE);
            } catch (IOException e) {
                serverLogger.error("Could not create log file: " + logFilePath + " " + e.getMessage(), e);
                System.exit(-1);
            }
        }
        return OutputStream.nullOutputStream();
    }

    private Domain runLevel(Path levelPath, ArgumentParser args, OutputStream logFileStream,
                            boolean hasGUIOutput, boolean closeLogOnExit) {
        serverLogger.info(String.format("Running client on level file: %s", levelPath.toString()));

        Domain domain = this.loadDomain(levelPath, false);
        if (domain == null) return null;

        PlaybackManager playbackManager = null;
        if (hasGUIOutput) {
            playbackManager = loadAndStartGUI(domain, args);
        } else {
            // The domain can discard past states if we don't need to be able to seek states in the GUI.
            domain.allowDiscardingPastStates();
        }

        Client client = loadAndStartClient(domain, args.getClientCommand(), args.getTimeoutSeconds(), logFileStream, closeLogOnExit);
        if (client == null) return null;

        if (playbackManager != null) {
            // FIXME: Holding class sync lock will not allow shutdown hook to signal GUI shutdown, deadlocking.
            playbackManager.waitShutdown();
            client.expireTimeout(); // Does nothing if timeout already expired or stopped.
        }

        client.waitShutdown();
        return domain;
    }

    private Domain loadDomain(Path levelPath, boolean isReplay) {
        serverLogger.info(String.format("Loading file : %s. Replay: %s", levelPath, isReplay));
        Domain domain = null;
        try {
            domain = Domain.loadLevel(levelPath, isReplay);
        } catch (ParseException e) {
            // TODO: Better error message (level invalid, rather than "failing" to parse).
            serverLogger.error("Could not load domain, failed to parse level file. " + e.getMessage(), e);
        } catch (IOException e) {
            serverLogger.error("IOException while loading domain. " + e.getMessage(), e);
        }
        return domain;
    }

    private Client loadAndStartClient(Domain domain, String clientCommand, int timeoutArg,
                                      OutputStream logFileStream, boolean closeLogOnExit) {
        Client client = null;
        try {
            long timeoutNS = timeoutArg * 1_000_000_000L;
            client = new Client(domain, clientCommand, logFileStream, closeLogOnExit, new Timeout(), timeoutNS);
            client.startProtocol();
        } catch (Exception e) {
            // TODO: Start writing errors to log file also? Will complicate/break parsing from domains though.
            serverLogger.error("Could not start client process. " + e.getMessage(), e);
            try {
                logFileStream.close();
            } catch (IOException e1) {
                clientLogger.error("Could not close log file. " + e1.getMessage(), e1);
            }
        }
        return client;
    }

    private PlaybackManager loadAndStartGUI(Domain domain, ArgumentParser args) {
        return this.loadAndStartGUI(new Domain[]{domain}, args);
    }

    private PlaybackManager loadAndStartGUI(Domain[] domains, ArgumentParser args) {
        serverLogger.debug("Loading GUI.");
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (ClassNotFoundException | UnsupportedLookAndFeelException |
                IllegalAccessException | InstantiationException e) {
            serverLogger.warn("Could not set system look and feel. " + e.getMessage(), e);
        }
        var gcs = getGraphicsConfigurations(domains.length, args.getScreens());
        PlaybackManager playbackManager = new PlaybackManager(domains, gcs);
        serverLogger.debug("Starting GUI.");
        playbackManager.startGUI(args.isStartFullscreen(), args.isStartHiddenInterface(),
                args.getMsPerAction(), args.isStartPlaying());
        playbackManager.focusPlaybackFrame(0);
        return playbackManager;
    }

    private DirectoryStream.Filter<Path> createLevelFilter() {
        return entry -> Files.isReadable(entry) &&
                Files.isRegularFile(entry) &&
                entry.getFileName().toString().endsWith(".lvl") &&
                entry.getFileName().toString().length() > 4;
    }

    private void createZipEntry(Path levelPath, ZipOutputStream logZipStream) {
        String levelFileName = levelPath.getFileName().toString();
        String logEntryName = levelFileName.substring(0, levelFileName.length() - 4) + ".log";
        try {
            logZipStream.putNextEntry(new ZipEntry(logEntryName));
        } catch (IOException e) {
            serverLogger.error("Could not create log file entry for level. " + e.getMessage(), e);
        }
    }

    private void writeSummary(ZipOutputStream logZipStream, ArrayList<Domain> domains) {
        try {
            logZipStream.putNextEntry(new ZipEntry("summary.txt"));
            BufferedWriter logWriter =
                    new BufferedWriter(new OutputStreamWriter(logZipStream, StandardCharsets.US_ASCII.newEncoder()));
            double solutionLength = 0, time = 0, memoryUsage = 0;
            int solved = 0;
            for (Domain domain : domains) {
                logWriter.write("Level name: ");
                logWriter.write(domain.getLevelName());
                logWriter.newLine();
                for (String statusLine : domain.getStatus()) {
                    logWriter.write(statusLine);
                    logWriter.newLine();
                }
                logWriter.newLine();
                logWriter.flush();

                if (domain instanceof Hospital2Domain) {
                    solutionLength += ((Hospital2Domain) domain).getNumActions();
                    time += (((Hospital2Domain) domain).getTime() / 1_000_000_000d);
                    memoryUsage += ((Hospital2Domain) domain).getMaxMemoryUsage();
                    if (((Hospital2Domain) domain).isSolved()) solved++;
                }
            }
            if (domains.stream().anyMatch(d -> d instanceof Hospital2Domain)) {
                int domainSize = domains.size();
                logWriter.write("Solved: ");
                logWriter.write(solved + "/" + domainSize);
                logWriter.newLine();
                logWriter.write("Average solution length: ");
                logWriter.write(String.format("%4.2f", solutionLength / domainSize));
                logWriter.newLine();
                logWriter.write("Average time: ");
                logWriter.write(String.format("%4.2f s", time / domainSize));
                logWriter.newLine();
                logWriter.write("Average memory usage: ");
                logWriter.write(String.format("%4.2f MB", memoryUsage / domainSize));
                logWriter.newLine();
                logWriter.flush();
            }
        } catch (IOException e) {
            serverLogger.error("Could not write summary to log file. " + e.getMessage(), e);
        }
    }

    private Domain[] loadReplayDomains(Path[] replayFilePaths) {
        Domain[] domains = new Domain[replayFilePaths.length];
        for (int i = 0; i < replayFilePaths.length; i++) {
            domains[i] = this.loadDomain(replayFilePaths[i], true);
            if (domains[i] == null) System.exit(-1);
        }
        return domains;
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
//        logger.debug("Shutdown begin");
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
//        logger.debug("Shutdown end");
//    }

    /**
     * Returns a GraphicsConfiguration array of size numScreens, attempting to map the user-specified
     * screen numbers (if provided) to the GraphicsDevices corresponding to a logical
     * left-to-right, top-to-bottom enumeration. See the screen notes in ArgumentParser.
     */
    private GraphicsConfiguration[] getGraphicsConfigurations(int numScreens, int[] screens) {
        final GraphicsConfiguration defaultScreen = GraphicsEnvironment
                .getLocalGraphicsEnvironment()
                .getDefaultScreenDevice()
                .getDefaultConfiguration();

        // WARNING! getScreenDevices() gives a direct reference to underlying array.
        // Copy before sorting, so we don't mutate underlying array.
        GraphicsDevice[] graphicsDevices = GraphicsEnvironment.getLocalGraphicsEnvironment().getScreenDevices();
        graphicsDevices = Arrays.copyOf(graphicsDevices, graphicsDevices.length);
        // Attempt sorting for the logical enumeration. May not work, e.g. if bounds are not relative to each other.
        Arrays.sort(graphicsDevices,
                Comparator.comparingInt((GraphicsDevice gd) -> gd.getDefaultConfiguration().getBounds().x)
                        .thenComparingInt(gd -> gd.getDefaultConfiguration().getBounds().y));

        var graphicsConfigurations = new GraphicsConfiguration[numScreens];
        for (int i = 0; i < numScreens; ++i) {
            if (i >= screens.length) {
                // Unspecified.
                graphicsConfigurations[i] = defaultScreen;
            } else if (screens[i] < 0 || screens[i] >= graphicsDevices.length) {
                // Out of range.
                graphicsConfigurations[i] = defaultScreen;
                serverLogger.warn("No screen #" + screens[i] + "; using default screen.");
            } else {
                // User-specified.
                graphicsConfigurations[i] = graphicsDevices[screens[i]].getDefaultConfiguration();
            }
        }

        return graphicsConfigurations;
    }
}
