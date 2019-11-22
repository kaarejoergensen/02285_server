package client;

import domain.Domain;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.*;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

public class Client {
    private Logger clientLogger = LogManager.getLogger("client");
    private Logger serverLogger = LogManager.getLogger("server");
    
    private Process clientProcess;
    private Thread clientThread;

    private final Timeout timeout;
    private long timeoutNS;

    private Domain domain;

    private BufferedInputStream clientIn;
    private BufferedOutputStream clientOut;
    private OutputStream logOut;

    private boolean closeLogOnExit;
    private boolean running = false;
    private boolean finished = false;

    /**
     * If the constructor succeeds, then the client process will already be spawned (so can't abort easily).
     * Otherwise, an IOException is thrown and the client process is not spawned.
     * <p>
     * If the constructor succeeds, then Client.startProtocol() has to be called for the Client Thread to terminate.
     * The Client Thread starts the Protocol Thread and has it run the Domain.runProtocol() function.
     * <p>
     * The Protocol Thread should communicate the timeouts to the Client Thread through the Timeout object that
     * it is passed as argument to Domain.runProtocol().
     * <p>
     * If the timeout expires before the Protocol Thread stops or extends the timeout, then the Client thread assumes
     * that the Protocol Thread is indefinitely blocked and proceeds to forcibly terminate the client process.
     * <p>
     * The Client Thread will finally wait for the Protocol Thread to exit and then itself exit.
     */
    public Client(Domain domain, String clientCommand, OutputStream logOut, boolean closeLogOnExit,
                  Timeout timeout, long timeoutNS) throws IOException {
        this.domain = domain;
        this.logOut = logOut;
        this.closeLogOnExit = closeLogOnExit;
        this.timeout = timeout;
        this.timeoutNS = timeoutNS;

        // Naïvely tokenize client command.
        ProcessBuilder processBuilder = new ProcessBuilder();
        processBuilder.command(Arrays.asList(clientCommand.strip().split("\\s++")));
        processBuilder.redirectError(ProcessBuilder.Redirect.INHERIT);

        this.clientProcess = processBuilder.start();

        InputStream clientIn = this.clientProcess.getInputStream();
        this.clientIn = clientIn instanceof BufferedInputStream ?
                (BufferedInputStream) clientIn :
                new BufferedInputStream(clientIn);
        OutputStream clientOut = this.clientProcess.getOutputStream();
        this.clientOut = clientOut instanceof BufferedOutputStream ?
                (BufferedOutputStream) clientOut :
                new BufferedOutputStream(clientOut);

        this.clientThread = new Thread(this::runClient, "ClientThread");
        this.clientThread.start();
    }

    public synchronized void startProtocol() {
        this.running = true;
        this.notifyAll();
    }

    public void waitShutdown() {
        synchronized (this) {
            while (!this.finished) {
                try {
                    this.wait();
                } catch (InterruptedException ignored) {
                }
            }
        }

        while (true) {
            try {
                this.clientThread.join();
                return;
            } catch (InterruptedException ignored) {
            }
        }
    }

    public void expireTimeout() {
        this.timeout.expire();
    }

    private void runClient() {
        clientLogger.debug("Thread started.");

        // Wait until startProtocol() called by Main Thread.
        synchronized (this) {
            while (!this.running) {
                try {
                    this.wait();
                } catch (InterruptedException ignored) {
                }
            }
        }

        clientLogger.debug(String.format("Client process supports normal termination: %s.",
                this.clientProcess.supportsNormalTermination()));

        // Start Protocol Thread.
        Thread protocolThread = new Thread(this::runProtocol, "ProtocolThread");
        protocolThread.start();

        // Wait for timeout to stop or expire. Handle accordingly.
        boolean timeoutExpired = this.timeout.waitTimeout();

        if (!timeoutExpired) {
            clientLogger.debug("ProtocolThread stopped timeout, waiting for client to terminate.");

            while (true) {
                try {
                    protocolThread.join();
                    break;
                } catch (InterruptedException ignored) {
                }
            }

            clientLogger.info("Waiting for client process to terminate by itself.");
            try {
                this.clientProcess.waitFor(500, TimeUnit.MILLISECONDS);
            } catch (InterruptedException ignored) {
            }

            this.terminateClient();

            // Close client process' streams.
            // FIXME: What happens if we didn't manage to terminate the client. Can closing the streams block this
            //  thread?
            this.closeClientStreams();
        } else {
            clientLogger.info("Client timed out.");

            this.terminateClient();

            // FIXME: What happens if we didn't manage to terminate the client. Can closing the streams block this
            //  thread?
            this.closeClientStreams();

            // FIXME: Should we time this out also, or can we expect all Domain implementations of runProtocol() to
            //  behave?
            //  If we leaked the client processes, then this could potentially also block indefinitely.
            while (true) {
                try {
                    protocolThread.join();
                    break;
                } catch (InterruptedException ignored) {
                }
            }
        }

        if (this.closeLogOnExit) {
            try {
                this.logOut.flush();
                this.logOut.close();
                clientLogger.debug("Closed log stream.");
            } catch (IOException e) {
                clientLogger.error("Could not flush and close log file. " + e.getMessage(), e);
                // FIXME: Handle or flag error back to Server?
            }
        }

        // Print status of run.
        synchronized (System.out) {
            for (String s : domain.getStatus()) {
                serverLogger.info(s);
            }
        }

        synchronized (this) {
            this.finished = true;
            this.notifyAll();
        }

        clientLogger.debug("Thread shut down.");
    }

    private void runProtocol() {
        clientLogger.debug("Thread started.");

        this.domain.runProtocol(this.timeout, this.timeoutNS, this.clientIn, this.clientOut, this.logOut);

        // If Domain.runProtocol() forgot to call Timeout.stop(), we call it here (does nothing if already stopped or
        // expired).
        this.timeout.stop();

        clientLogger.debug("Thread shut down.");
    }

    private void terminateClient() {
        if (this.clientProcess.isAlive() && this.clientProcess.supportsNormalTermination()) {
            clientLogger.info("Sending termination signal to client process (PID = " + this.clientProcess.pid() + ").");
            this.clientProcess.destroy();
            try {
                boolean terminated = this.clientProcess.waitFor(1000, TimeUnit.MILLISECONDS);
                if (terminated) {
                    return;
                }
            } catch (InterruptedException ignored) {
            }
        }

        if (this.clientProcess.isAlive()) {
            clientLogger.info("Forcibly terminating client process.");
            this.clientProcess.destroyForcibly();
            try {
                boolean terminated = this.clientProcess.waitFor(200, TimeUnit.MILLISECONDS);
                if (terminated) {
                    return;
                }
            } catch (InterruptedException ignored) {
            }
        }

        var clientChildProcesses = this.clientProcess.descendants()
                .filter(ProcessHandle::isAlive)
                .collect(Collectors.toSet());
        if (this.clientProcess.isAlive()) {
            clientLogger.warn("Client process not terminated. PID = " + this.clientProcess.pid() + ".");
        } else if (!clientChildProcesses.isEmpty()) {
            synchronized (System.out) {
                clientLogger.warn("Client spawned subprocesses which haven't terminated.");
                String leakedPIDs = clientChildProcesses.stream()
                        .map(ph -> Long.toString(ph.pid()))
                        .collect(Collectors.joining(", "));
                clientLogger.warn("PIDs: " + leakedPIDs + ".");
            }
        } else {
            clientLogger.info("Client terminated.");
        }
    }

    private void closeClientStreams() {
        try {
            this.clientIn.close();
            this.clientOut.close();
            this.clientProcess.getErrorStream().close();
        } catch (IOException ignored) { }
    }
}
