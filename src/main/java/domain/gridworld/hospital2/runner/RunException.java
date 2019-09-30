package domain.gridworld.hospital2.runner;

public class RunException extends Exception {
    public RunException(String message) {
        super(message);
    }

    public RunException(String message, Throwable cause) {
        super(message, cause);
    }
}
