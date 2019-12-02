package searchclient.nn.impl;

import org.javatuples.Pair;
import searchclient.State;
import searchclient.nn.NNet;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;

public class PythonNNet extends NNet {
    private static final String PYTHON_PATH = "./venv/bin/python";
    private static final String SCRIPT_PATH = "./src/main/python/Main.py";

    private Process process;
    private BufferedReader clientReader;
    private BufferedWriter clientWriter;

    public PythonNNet() throws IOException {
        this.run();
    }

    private void run() throws IOException {
        Runtime.getRuntime().addShutdownHook(new Thread(this::shutdownPython));
        ProcessBuilder processBuilder = new ProcessBuilder(PYTHON_PATH, SCRIPT_PATH).redirectError(ProcessBuilder.Redirect.INHERIT);
        this.process = processBuilder.start();

        InputStream clientIn = process.getInputStream();
        BufferedInputStream outputFromClient = clientIn instanceof BufferedInputStream ?
                (BufferedInputStream) clientIn :
                new BufferedInputStream(clientIn);
        OutputStream clientOut = process.getOutputStream();
        BufferedOutputStream inputToClient = clientOut instanceof BufferedOutputStream ?
                (BufferedOutputStream) clientOut :
                new BufferedOutputStream(clientOut);

        this.clientReader = new BufferedReader(new InputStreamReader(outputFromClient, StandardCharsets.UTF_8));
        this.clientWriter = new BufferedWriter(new OutputStreamWriter(inputToClient, StandardCharsets.UTF_8));
    }

    private void shutdownPython() {
        this.process.destroy();
    }

    @Override
    public float train(Pair<List<String>, List<Double>> trainingSet) {
        this.writeToPython("train", trainingSet.getValue0().toString()
                + System.lineSeparator() + trainingSet.getValue1().toString());
        return Float.parseFloat(this.readFromPython());
    }

    @Override
    public float predict(State state) {
        this.writeToPython("predict", state.toMLString());
        return Float.parseFloat(this.readFromPython());
    }

    @Override
    public void saveModel(Path fileName) {
        this.writeToPython("saveModel", fileName.toString());
        this.readFromPython();
    }

    @Override
    public void loadModel(Path fileName) {
        this.writeToPython("loadModel", fileName.toString());
        this.readFromPython();
    }

    @Override
    public void close() throws IOException {
        this.writeToPython("close", "");
        this.readFromPython();
        this.clientWriter.close();
        this.clientReader.close();
        try {
            Thread.sleep(200);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        if (this.process.isAlive()) this.process.destroy();
    }

    private String readFromPython() {
        String line = "";
        synchronized (this) {
            try {
                line = this.clientReader.readLine();
                if (line == null) {
                    System.err.println("Read from python failed: null line received");
                    System.exit(-1);
                }
            } catch (IOException e) {
                System.err.println("Read from python failed: " + e.getMessage());
                System.exit(-1);
            }
        }
        return line;
    }

    private void writeToPython(String method, String args) {
        synchronized (this) {
            try {
                this.clientWriter.write(method + System.lineSeparator());
                this.clientWriter.flush();
                this.clientWriter.write(args + System.lineSeparator());
                this.clientWriter.flush();
                this.clientWriter.write("done" + System.lineSeparator());
                this.clientWriter.flush();
            } catch (IOException e) {
                System.err.println("Write to python failed: " + e.getMessage());
                System.exit(-1);
            }
        }
    }
}
