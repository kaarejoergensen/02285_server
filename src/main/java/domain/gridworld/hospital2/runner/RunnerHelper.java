package domain.gridworld.hospital2.runner;

import client.Timeout;
import domain.gridworld.hospital2.state.State;
import domain.gridworld.hospital2.state.objects.StaticState;
import lombok.RequiredArgsConstructor;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.javatuples.Pair;
import server.CustomLoggerConfigFactory;
import shared.Action;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.CharacterCodingException;
import java.util.List;

@RequiredArgsConstructor
class RunnerHelper {
    private static Logger clientLogger = LogManager.getLogger("client");

    static String readClientName(BufferedReader clientReader, Timeout timeout, long startNS, long timeoutNS) throws RunException {
        String clientMsg;
        try {
            clientLogger.debug("Waiting for client name.");
            clientMsg = clientReader.readLine();
        } catch (CharacterCodingException e) {
            throw new RunException("Client message not valid ASCII.");
        } catch (IOException e) {
            // FIXME: What may even cause this? Closing the stream causes readLine to return null rather than throw.
            synchronized (System.out) {
                throw new RunException("Unexpected exception while reading client name:", e);
            }
        }

        // Check and reset timeout.
        if (!timeout.reset(startNS, timeoutNS)) {
            throw new RunException("Timed out while waiting for client name.");
        }

        // Store client name.
        if (clientMsg != null) {
            clientLogger.debug("Received client name: " + clientMsg);
            return clientMsg;
        } else {
            throw new RunException("Client closed its output stream before sending its name.");
        }
    }

    static void writeLevelToClientAndLog(BufferedWriter clientWriter, BufferedWriter logWriter,
                                         String level, String clientName, Timeout timeout) throws RunException {
        clientLogger.debug("Writing level to client and log.");
        try {
            clientWriter.write(level);
            logWriter.write(level);
            clientWriter.flush();
            logWriter.flush();
        } catch (IOException e) {
            if (timeout.isExpired()) {
                throw new RunException("Timeout expired while sending level to client.");
            } else {
                throw new RunException("Could not send level to client and/or log. " + e.getMessage(), e);
            }
        }

        // Log client name.
        // Has to be logged after level file contents have been logged (first log lines must be domain).
        try {
            clientLogger.debug("Logging client name.");
            logWriter.write("#clientname" + System.lineSeparator());
            logWriter.write(clientName+ System.lineSeparator());
            logWriter.flush();
        } catch (IOException e) {
            throw new RunException("Could not write client name to log file. " + e.getMessage(), e);
        }
    }

    static Pair<Long, Double> exchangeActions(BufferedReader clientReader, BufferedWriter clientWriter, BufferedWriter logWriter,
                                StaticState staticState, Timeout timeout, long startNS, List<State> states, boolean allowDiscardingPastStates) throws RunException {
        clientLogger.debug("Beginning action/comment message exchanges.");
        long numMessages = 0, numActions = 0;
        double maxMemoryUsage = 0;
        Action[] jointAction = new Action[staticState.getNumAgents()];
        String clientMsg;
        try {
            logWriter.write("#actions" + System.lineSeparator());
            logWriter.flush();
        } catch (IOException e) {
            throw new RunException("Could not write to log file. " + e.getMessage(), e);
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
                throw new RunException("Client message not valid ASCII.");
            } catch (IOException e) {
                throw new RunException("Unexpected exception while reading from client. " + e.getMessage(), e);
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
                if (clientMsg.startsWith("#memory")) {
                    String[] memoryMsg = clientMsg.split(" ");
                    double memory = Double.parseDouble(memoryMsg[1]);
                    if (memory > maxMemoryUsage)
                        maxMemoryUsage = memory;
                } else {
                    clientLogger.log(CustomLoggerConfigFactory.messageLevel, clientMsg.substring(1));
                }
            } else {
                // Parse action string.
                String[] actionMsg = clientMsg.split(";");
                if (actionMsg.length != staticState.getNumAgents()) {
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
                boolean[] result = execute(jointAction, actionTime, states, staticState, allowDiscardingPastStates);
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
                    throw new RunException("Could not write response to client. " + e.getMessage(), e);
                }

                // Log action.
                try {
                    logWriter.write(actionTime + ":" + clientMsg + System.lineSeparator());
                    logWriter.flush();
                } catch (IOException e) {
                    throw new RunException("Could not write to log file. " + e.getMessage(), e);
                }
                numActions++;
            }
        }
        clientLogger.debug("Messages exchanged: " + numMessages + ".");
        return Pair.with(numActions, maxMemoryUsage);
    }

    /**
     * Execute a joint action.
     * Returns a boolean array with success for each agent.
     */
    static boolean[] execute(Action[] jointAction, long actionTime, List<State> states,
                             StaticState staticState, boolean allowDiscardingPastStates) {
        State state = states.get(states.size() - 1);

        // Create new state with applicable and non-conflicting actions.
        Pair<State, boolean[]> result = state.apply(jointAction, staticState);
        result.getValue0().setStateTime(actionTime);

        if (allowDiscardingPastStates) {
            states.clear();
        }
        states.add(result.getValue0());

        return result.getValue1();
    }

    static void writeLogSummary(BufferedWriter logWriter, boolean solved, long numActions, long time, double memory) throws RunException {
        try {
            logWriter.write("#end" + System.lineSeparator());
            logWriter.write("#solved" + System.lineSeparator());
            logWriter.write((solved ? "true" : "false") + System.lineSeparator());
            logWriter.write("#numactions" + System.lineSeparator());
            logWriter.write(numActions + System.lineSeparator());
            logWriter.write("#time" + System.lineSeparator());
            logWriter.write(time + System.lineSeparator());
            logWriter.write("#memory" + System.lineSeparator());
            logWriter.write(String.format("%4.2f MB", memory) + System.lineSeparator());
            logWriter.write("#end" + System.lineSeparator());
            logWriter.flush();
        } catch (IOException e) {
            throw new RunException("Could not write to log file. " + e.getMessage(), e);
        }
    }
}
