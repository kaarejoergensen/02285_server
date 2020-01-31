package searchclient.nn.impl;

import lombok.Getter;
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
    private static final String SCRIPT_PATH = "./src/main/python/Main.py";

    private static final String TEMP_PATH = "models/";
    private static final String TEMP_NAME = "temp.pth";

    @Getter private String pythonPath;
    @Getter private Integer gpu;
    @Getter private Float lr;
    @Getter private Integer epochs;
    @Getter private Integer batchSize;
    @Getter private Integer resBlocks;
    @Getter private String lossFunction;
    @Getter private Integer features;

    private Process process;
    private BufferedReader clientReader;
    private BufferedWriter clientWriter;

    public PythonNNet(String pythonPath, Integer gpu, Float lr, Integer epochs, Integer batchSize, Integer resBlocks, String lossFunction, Integer features) throws IOException {
        this.pythonPath = pythonPath;
        this.gpu = gpu;
        this.lr = lr;
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.resBlocks = resBlocks;
        this.lossFunction = lossFunction;
        this.features = features;
        this.run();
    }

    public PythonNNet(PythonNNet pythonNNet, int gpu) throws IOException {
        this.pythonPath = pythonNNet.getPythonPath();
        this.gpu = gpu;
        this.lr = pythonNNet.lr;
        this.epochs = pythonNNet.epochs;
        this.batchSize = pythonNNet.batchSize;
        this.resBlocks = pythonNNet.resBlocks;
        this.lossFunction = pythonNNet.lossFunction;
        this.features = pythonNNet.features;
        this.run();
    }

    private void run() throws IOException {
        Runtime.getRuntime().addShutdownHook(new Thread(this::shutdownPython));
        ProcessBuilder processBuilder = new ProcessBuilder(this.pythonPath, SCRIPT_PATH).redirectError(ProcessBuilder.Redirect.INHERIT);
        if (gpu != null) {
            processBuilder.command().add("--gpu");
            processBuilder.command().add(gpu.toString());
        }
        if (lr != null) {
            processBuilder.command().add("--lr");
            processBuilder.command().add(lr.toString());
        }
        if (epochs != null) {
            processBuilder.command().add("--epochs");
            processBuilder.command().add(epochs.toString());
        }
        if (batchSize != null) {
            processBuilder.command().add("--batch_size");
            processBuilder.command().add(batchSize.toString());
        }
        if (resBlocks != null) {
            processBuilder.command().add("--resblocks");
            processBuilder.command().add(resBlocks.toString());
        }
        if (lossFunction != null) {
            processBuilder.command().add("--loss_function");
            processBuilder.command().add(lossFunction);
        }if (features != null) {
            processBuilder.command().add("--features");
            processBuilder.command().add(features.toString());
        }

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
        return this.readPredictResult(clientReader);
    }

    private PredictResult readPredictResult(BufferedReader clientReader) {
        double[] probabilityVector = parseProbabilityVector(this.readFromPython());
        float score = Float.parseFloat(this.readFromPython().replaceAll("[\\[\\]]", ""));
        return new PredictResult(probabilityVector, score);
    }

    private static double[] parseProbabilityVector(String string) {
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
            return new PythonNNet(this.pythonPath, this.gpu, this.lr, this.epochs, this.batchSize, this.resBlocks, this.lossFunction, this.features);
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
        return line;
    }

    private void writeToPython(String method, String args) {
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
