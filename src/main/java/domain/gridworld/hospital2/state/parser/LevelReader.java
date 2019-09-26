package domain.gridworld.hospital2.state.parser;

import java.io.IOException;
import java.io.LineNumberReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

public class LevelReader implements AutoCloseable {
    private LineNumberReader levelReader;
    private StringBuilder levelStringBuilder;

    LevelReader(Path domainFile) throws IOException {
        this.levelReader = new LineNumberReader(Files.newBufferedReader(domainFile,
                StandardCharsets.US_ASCII));
        levelStringBuilder = new StringBuilder();
    }

    String readLine() throws IOException {
        String line = this.levelReader.readLine();
        if (line != null) levelStringBuilder.append(line).append(System.lineSeparator());
        return line;
    }

    @Override
    public void close() throws IOException {
        this.levelReader.close();
    }

    int getLineNumber() {
        return this.levelReader.getLineNumber();
    }

    public String getLevel() {
        if (!this.levelStringBuilder.toString().endsWith("\n")) {
            this.levelStringBuilder.append(System.lineSeparator());
        }
        return this.levelStringBuilder.toString();
    }
}
