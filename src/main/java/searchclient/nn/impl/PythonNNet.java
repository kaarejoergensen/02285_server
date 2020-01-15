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

    private String pythonPath = PYTHON_PATH;

    private Process process;
    private BufferedReader clientReader;
    private BufferedWriter clientWriter;

    public PythonNNet() throws IOException {
        this.run();
    }

    public PythonNNet(String pythonPath) throws IOException {
        this.pythonPath = pythonPath;
        this.run();
    }

    private void run() throws IOException {
        Runtime.getRuntime().addShutdownHook(new Thread(this::shutdownPython));
        ProcessBuilder processBuilder = new ProcessBuilder(this.pythonPath, SCRIPT_PATH).redirectError(ProcessBuilder.Redirect.INHERIT);
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
        writeToPython("train", trainingSet.stream().collect(Collectors.joining(System.lineSeparator())), clientWriter);
        return Float.parseFloat(readFromPython(clientReader));
    }

    @Override
    public PredictResult predict(State state) {
        writeToPython("predict", state.toMLString(), clientWriter);
        return readPredictResult(clientReader);
    }

    private static synchronized PredictResult readPredictResult(BufferedReader clientReader) {
        double[] probabilityVector = parseProbabilityVector(readFromPython(clientReader));
        float score = Float.parseFloat(readFromPython(clientReader).replaceAll("[\\[\\]]", ""));
        float loss = Float.parseFloat(readFromPython(clientReader));
        return new PredictResult(probabilityVector, score, loss);
    }

    private static double[] parseProbabilityVector(String string) {
        return Arrays.stream(string.replaceAll("[\\[\\] ]", "").split(","))
                .mapToDouble(Double::parseDouble).toArray();
    }

    @Override
    public void saveModel(Path fileName) {
        writeToPython("saveModel", fileName.toString(), clientWriter);
        readFromPython(clientReader);
    }

    @Override
    public void loadModel(Path fileName) {
        writeToPython("loadModel", fileName.toString(), clientWriter);
        readFromPython(clientReader);
    }

    public void saveTempModel() throws IOException {
        if (!Files.isDirectory(Path.of(TEMP_PATH))) {
            Files.createDirectories(Path.of(TEMP_PATH));
        }
        Path temp = Path.of(TEMP_PATH + TEMP_NAME);
        this.saveModel(temp);
    }

    public void loadTempModel() {
        Path temp = Path.of(TEMP_PATH + TEMP_NAME);
        this.loadModel(temp);
    }

    @Override
    public NNet clone() {
        try {
            return new PythonNNet();
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
        writeToPython("close", "", clientWriter);
        readFromPython(clientReader);
        this.clientWriter.close();
        this.clientReader.close();
        try {
            Thread.sleep(200);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        if (this.process.isAlive()) this.process.destroy();
    }

    private static synchronized String readFromPython(BufferedReader clientReader) {
        String line = "";
        try {
            line = clientReader.readLine();
            if (line == null) {
                System.err.println("Read from python failed: null line received");
                System.exit(-1);
            }
        } catch (IOException e) {
            System.err.println("Read from python failed: " + e.getMessage());
            System.exit(-1);
        }
        return line;
    }

    private static synchronized void writeToPython(String method, String args, BufferedWriter clientWriter) {
        try {
            clientWriter.write(method + System.lineSeparator());
            clientWriter.flush();
            clientWriter.write(args + System.lineSeparator());
            clientWriter.flush();
            clientWriter.write("done" + System.lineSeparator());
            clientWriter.flush();
        } catch (IOException e) {
            System.err.println("Write to python failed: " + e.getMessage());
            System.exit(-1);
        }
    }
}
