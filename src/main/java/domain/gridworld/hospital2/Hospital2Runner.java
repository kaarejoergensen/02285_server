package domain.gridworld.hospital2;

import client.Timeout;
import lombok.Getter;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import server.CustomLoggerConfigFactory;

import java.io.*;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;

public class Hospital2Runner {
    private Logger clientLogger = LogManager.getLogger("client");

    private BufferedReader clientReader;
    private BufferedWriter clientWriter, logWriter;
    private Timeout timeout;
    private long timeoutNS, startNS;
    private String level;
    private Hospital2Domain domain;

    @Getter private String clientName;
    @Getter private long numActions;

    public Hospital2Runner(Timeout timeout, long timeoutNS, String level, BufferedInputStream clientIn,
                           BufferedOutputStream clientOut, OutputStream logOut, Hospital2Domain domain) {
        this.timeout = timeout;
        this.timeoutNS = timeoutNS;
        this.level = level;

        this.clientReader = new BufferedReader(new InputStreamReader(clientIn, StandardCharsets.US_ASCII.newDecoder()));
        this.clientWriter = new BufferedWriter(new OutputStreamWriter(clientOut, StandardCharsets.US_ASCII.newEncoder()));
        this.logWriter = new BufferedWriter(new OutputStreamWriter(logOut, StandardCharsets.US_ASCII.newEncoder()));
        this.domain = domain;
    }

    public void run() {
        this.startNS = System.nanoTime();
        this.timeout.reset(startNS, TimeUnit.SECONDS.toNanos(10));

        this.readClientName();
        this.writeLevelToClientAndLog();
        this.exchangeActions();
        this.writeLogSummary();
        clientLogger.debug("Protocol finished.");
    }

    private void readClientName() {
        String clientMsg = null;
        try {
            clientLogger.debug("Waiting for client name.");
            clientMsg = clientReader.readLine();
        } catch (CharacterCodingException e) {
            clientLogger.error("Client message not valid ASCII.");
            System.exit(-1);
        } catch (IOException e) {
            // FIXME: What may even cause this? Closing the stream causes readLine to return null rather than throw.
            synchronized (System.out) {
                clientLogger.error("Unexpected exception while reading client name:");
                e.printStackTrace(System.out);
            }
            System.exit(-1);
        }

        // Check and reset timeout.

        if (!timeout.reset(startNS, timeoutNS)) {
            clientLogger.error("Timed out while waiting for client name.");
            System.exit(-1);
        }

        // Store client name.
        if (clientMsg != null) {
            clientLogger.debug("Received client name: " + clientMsg);
            this.clientName = clientMsg;
        } else {
            clientLogger.error("Client closed its output stream before sending its name.");
            System.exit(-1);
        }
    }

    private void writeLevelToClientAndLog() {
        clientLogger.debug("Writing level to client and log.");
        try {
            clientWriter.write(level);
            logWriter.write(level);
            clientWriter.flush();
            logWriter.flush();
        } catch (IOException e) {
            if (timeout.isExpired()) {
                clientLogger.error("Timeout expired while sending level to client.");
            } else {
                clientLogger.error("Could not send level to client and/or log. " + e.getMessage(), e);
            }
            System.exit(-1);
        }

        // Log client name.
        // Has to be logged after level file contents have been logged (first log lines must be domain).
        try {
            clientLogger.debug("Logging client name.");
            logWriter.write("#clientname" + System.lineSeparator());
            logWriter.write(this.clientName+ System.lineSeparator());
            logWriter.flush();
        } catch (IOException e) {
            clientLogger.error("Could not write client name to log file. " + e.getMessage(), e);
            System.exit(-1);
        }
    }

    private void exchangeActions() {
        clientLogger.debug("Beginning action/comment message exchanges.");
        long numMessages = 0;
        Action[] jointAction = new Action[this.domain.getNumAgents()];
        String clientMsg;
        try {
            logWriter.write("#actions");
            logWriter.newLine();
            logWriter.flush();
        } catch (IOException e) {
            clientLogger.error("Could not write to log file. " + e.getMessage(), e);
            return;
        }

        protocolLoop:
        while (true) {
            if (timeout.isExpired()) {
                clientLogger.debug("Client timed out in protocol loop.");
                break;
            }

            // Read client message.
            try {
                clientMsg = clientReader.readLine();
                clientLogger.debug("Message from client: " + clientMsg);
            } catch (CharacterCodingException e) {
                clientLogger.error("Client message not valid ASCII.");
                return;
            } catch (IOException e) {
                clientLogger.error("Unexpected exception while reading from client. " + e.getMessage(), e);
                return;
            }
            if (clientMsg == null) {
                if (timeout.isExpired()) {
                    clientLogger.debug("Client stream closed after timeout.");
                } else {
                    clientLogger.debug("Client closed its output stream.");
                }
                break;
            }

            if (timeout.isExpired()) {
                clientLogger.debug("Client timed out in protocol loop.");
                break;
            }

            // Process message.
            ++numMessages;
            if (clientMsg.startsWith("#")) {
                clientLogger.log(CustomLoggerConfigFactory.messageLevel, clientMsg.substring(1));
            } else {
                // Parse action string.
                String[] actionMsg = clientMsg.split(";");
                if (actionMsg.length != this.domain.getNumAgents()) {
                    clientLogger.error("Invalid number of agents in joint action:");
                    clientLogger.error(clientMsg);
                    continue;
                }
                for (int i = 0; i < jointAction.length; ++i) {
                    jointAction[i] = Action.parse(actionMsg[i]);
                    if (jointAction[i] == null) {
                        clientLogger.error("Invalid joint action:");
                        clientLogger.error(clientMsg);
                        continue protocolLoop;
                    }
                }

                // Execute action.
                long actionTime = System.nanoTime() - startNS;
                boolean[] result = this.domain.execute(jointAction, actionTime);
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
                    clientLogger.error("Could not write response to client. " + e.getMessage(), e);
                    return;
                }

                // Log action.
                try {
                    logWriter.write(Long.toString(actionTime));
                    logWriter.write(":");
                    logWriter.write(clientMsg + System.lineSeparator());
                    logWriter.flush();
                } catch (IOException e) {
                    clientLogger.error("Could not write to log file. " + e.getMessage(), e);
                    return;
                }
            }
        }
        clientLogger.debug("Messages exchanged: " + numMessages + ".");
    }

    private void writeLogSummary() {
        try {
            logWriter.write("#end" + System.lineSeparator());

            logWriter.write("#solved" + System.lineSeparator());
            logWriter.write(this.domain.isSolved(this.domain.getLatestState()) ? "true" : "false" + System.lineSeparator());

            logWriter.write("#numactions" + System.lineSeparator());
            logWriter.write(Long.toString(this.numActions) + System.lineSeparator());

            logWriter.write("#time" + System.lineSeparator());
            logWriter.write(this.domain.getLatestState().getStateTime() + System.lineSeparator());

            logWriter.write("#end" + System.lineSeparator());
            logWriter.flush();
        } catch (IOException e) {
            clientLogger.error("Could not write to log file. " + e.getMessage(), e);
            System.exit(-1);
        }
    }
}
