package searchclient.nn.impl;

import searchclient.State;
import searchclient.nn.NNet;
import searchclient.nn.PredictResult;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class PythonNNet extends NNet {
    private static final String PYTHON_PATH = "./venv/bin/python";
    private static final String SCRIPT_PATH = "./src/main/python/Main.py";

    private static final String TEMP_PATH = "models/";
    private static final String TEMP_NAME = "temp.pth";

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
    public float train(List<String> trainingSet) {
        this.writeToPython("train", trainingSet.stream().collect(Collectors.joining(System.lineSeparator())));
        return Float.parseFloat(this.readFromPython());
    }

    @Override
    public PredictResult predict(State state) {
        this.writeToPython("predict", state.toMLString());
        double[] probabilityVector = this.parseProbabilityVector(this.readFromPython());
        float score = Float.parseFloat(this.readFromPython().replaceAll("[\\[\\]]", ""));
        float loss = Float.parseFloat(this.readFromPython());
        return new PredictResult(probabilityVector, score, loss);
    }

    private double[] parseProbabilityVector(String string) {
        return Arrays.stream(string.replaceAll("[\\[\\] ]", "").split(","))
                .mapToDouble(Double::parseDouble).toArray();
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
    public NNet clone() {
        try {
            if (!Files.isDirectory(Path.of(TEMP_PATH))) {
                Files.createDirectories(Path.of(TEMP_PATH));
            }
            Path temp = Path.of(TEMP_PATH + TEMP_NAME);
            this.saveModel(temp);
            NNet nnet = new PythonNNet();
            nnet.loadModel(temp);
            return nnet;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    @Override
    public String toString() {
        return "PNNET";
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
